import itertools
import logging
import sys
import math
import os
import warnings
from collections.abc import Iterable
from collections import Counter

from abc import ABCMeta, abstractmethod

import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle as dataset_shuffle
import finetune
from finetune.errors import FinetuneError
from finetune.encoding.input_encoder import EncodedOutput, tokenize_context
from finetune.util.imbalance import compute_class_weights
from finetune.util.input_utils import (
    InputMode,
    validation_settings,
    wrap_tqdm,
    Chunker,
    has_targets,
    batch_dataset,
)

LOGGER = logging.getLogger("finetune")


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        self._text_encoder = None
        self.label_encoder = None
        self.target_dim = None
        self.pad_idx_ = None
        self.rebuild = False
        self._chunker = None
        self.current_epoch_offset = 0
        self.total_epoch_offset = 0

    @property
    def text_encoder(self):
        if not hasattr(self, "_text_encoder") or self._text_encoder is None:
            self._text_encoder = self.config.base_model.get_encoder(self.config)
        return self._text_encoder

    @property
    def dataset_size(self):
        return self.config.dataset_size

    @abstractmethod
    def _target_encoder(self):
        # Overridden by subclass to produce the right target encoding for a given target model.
        raise NotImplementedError

    @property
    def chunker(self):
        if (
            getattr(self, "_chunker", None) is None
            or self.config.max_length != self._chunker.max_length
        ):
            self._chunker = Chunker(
                max_length=self.config.max_length,
                total_context_width=self.config.chunk_context,
                justify=self.config.chunk_alignment,
            )
        return self._chunker

    def _add_context_info_if_present(self, types, shapes):
        if self.config.use_auxiliary_info:
            TS = tf.TensorShape
            types["context"] = tf.float32
            shapes["context"] = TS([None, self.config.context_dim])
        return types, shapes

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {
            "tokens": TS([None]),
        }
        types, shapes = self._add_context_info_if_present(types, shapes)
        return (
            (types, tf.float32,),
            (shapes, TS([self.target_dim]),),
        )

    def zip_list_to_dict(self, X, Y=None, context=None):
        if Y is not None:
            Y = list(Y)
            if len(X) != len(Y):
                raise FinetuneError(
                    "the length of your labels does not match the length of your text"
                )
        if context is not None:
            context = list(context)
            if len(X) != len(context):
                raise FinetuneError(
                    "the length of your context does not match the length of your text"
                )
        out = []
        for i, x in enumerate(X):
            sample = {"X": x}
            if Y is not None:
                sample["Y"] = Y[i]
            if context is not None:
                sample["context"] = context[i]
            out.append(sample)
        return out

    def text_to_tokens_mask(self, X, Y=None, context=None):
        out_gen = self._text_to_ids(X, pad_token=self.config.pad_token)
        for i, out in enumerate(out_gen):
            if context is None:
                feats = {"tokens": out.token_ids}
            else:
                tokenized_context = tokenize_context(context, out, self.config)
                feats = {"tokens": out.token_ids, "context": tokenized_context}
            if Y is None:
                yield feats
            else:
                yield feats, self.label_encoder.transform([Y])[0]

    def _post_data_initialization(self, dataset=None):
        if "Y" in dataset[0]:
            ys = [data["Y"] for data in dataset]
            if self.label_encoder is None:
                self.label_encoder = self._target_encoder()
                self.label_encoder.fit(ys)

            self.config.pad_idx = self.pad_idx

            target_dim = self.label_encoder.target_dim
            self.lm_loss_coef = (
                self.config.lm_loss_coef if target_dim is not None else 1.0
            )
            self.target_dim = target_dim

    def _compute_class_counts(self, encoded_dataset):
        target_arrs = np.asarray([target_arr for doc, target_arr in encoded_dataset])
        targets = []
        for target in self.label_encoder.inverse_transform(target_arrs):
            if isinstance(target, Iterable):
                # Iterable
                targets.extend(target)
            else:
                targets.append(target)

        return Counter(targets)

    def _compute_class_weights(self, class_weights, class_counts):
        return compute_class_weights(
            class_weights=class_weights, class_counts=class_counts
        )

    def make_dataset_fn(
        self, data_fn, tqdm_mode, shapes, types, update_hook=None, skip_val=False
    ):
        def dataset_fn():
            return Dataset.from_generator(
                wrap_tqdm(
                    gen=data_fn,
                    mode=tqdm_mode,
                    n_epochs=self.config.n_epochs,
                    val_size=self.config.val_size,
                    dataset_size=self.config.dataset_size,
                    skip_val=skip_val,
                    quiet=self.config.debugging_logs,
                    update_hook=update_hook,
                    current_epoch_offset=self.current_epoch_offset if tqdm_mode == "train" else 0,
                    total_epoch_offset=self.total_epoch_offset if tqdm_mode == "train" else 0,
                ),
                types,
                shapes,
            )

        return dataset_fn

    def get_dataset_from_generator(self, generator_fn, input_mode, update_hook=None):
        def chunked_and_tokenized_dataset():
            for d in generator_fn():
                yield from self.text_to_tokens_mask(**d)

        types, shapes = self.feed_shape_type_def()

        if input_mode == InputMode.PREDICT:
            tqdm_mode = "predict"
        else:
            tqdm_mode = "train"

        if input_mode == InputMode.PREDICT or not has_targets(generator_fn):
            types = types[0]
            shapes = shapes[0]

        raw_dataset = self.make_dataset_fn(
            data_fn=chunked_and_tokenized_dataset,
            tqdm_mode=tqdm_mode,
            update_hook=update_hook,
            types=types,
            shapes=shapes,
            skip_val=input_mode == InputMode.TRAIN,
        )
        if input_mode == InputMode.PREDICT:
            return {
                "predict_dataset": batch_dataset(
                    raw_dataset,
                    batch_size=self.config.predict_batch_size,
                    shapes=shapes,
                )
            }

        if self.config.chunk_long_sequences:
            LOGGER.warning(
                "The dataset size is not adjusted for chunk long sequences when training from a generator"
            )

        if self.config.dataset_size is None:
            raise FinetuneError(
                "If you are using a callable as input you must provide config.dataset_size"
            )

        if self.config.class_weights is not None or self.config.oversample:
            raise FinetuneError(
                "Cannot use class weights or resampling in generator mode"
            )

        self.config.val_size, self.config.val_interval = validation_settings(
            dataset_size=self.config.dataset_size,
            batch_size=self.config.batch_size,
            val_size=self.config.val_size,
            val_interval=self.config.val_interval,
            keep_best_model=self.config.keep_best_model,
        )

        self.config.dataset_size -= self.config.val_size

        val_dataset = (
            lambda: raw_dataset()
            .shuffle(
                self.config.shuffle_buffer_size,
                seed=self.config.seed,
                reshuffle_each_iteration=False,
            )
            .take(self.config.val_size)
        )
        train_dataset = (
            lambda: raw_dataset()
            .shuffle(
                self.config.shuffle_buffer_size,
                seed=self.config.seed,
                reshuffle_each_iteration=False,
            )
            .skip(self.config.val_size)
        )

        return {
            "train_dataset": batch_dataset(
                train_dataset,
                batch_size=self.config.batch_size,
                shapes=shapes,
                n_epochs=self.config.n_epochs,
            ),
            "val_dataset": batch_dataset(
                val_dataset, batch_size=self.config.batch_size, shapes=shapes
            ),
        }

    def get_dataset_from_list(self, data_list, input_mode, update_hook=None):
        assert input_mode == InputMode.TRAIN, "use the generator path for prediction"

        data_list = list(data_list)
        self._post_data_initialization(data_list)

        self.config.val_size, self.config.val_interval = validation_settings(
            dataset_size=len(data_list),
            batch_size=self.config.batch_size,
            val_size=self.config.val_size,
            val_interval=self.config.val_interval,
            keep_best_model=self.config.keep_best_model,
        )

        if self.config.val_size > 0 and self.config.val_set is None:
            train_split, val_split = train_test_split(
                data_list, test_size=self.config.val_size, random_state=self.config.seed
            )
        else:
            train_split = dataset_shuffle(data_list, random_state=self.config.seed)
            val_split = self.config.val_set or []

        tokenized_train_split = list(
            itertools.chain.from_iterable(
                self.text_to_tokens_mask(**d) for d in train_split
            )
        )

        self.config.dataset_size = len(tokenized_train_split)

        tokenized_val_split = list(
            itertools.chain.from_iterable(
                self.text_to_tokens_mask(**d) for d in val_split
            )
        )
        if self.config.val_size != len(tokenized_val_split):
            LOGGER.warning(
                "Updating validation size from {} to {} this is possibly due to chunking but may cause issues with val frequency.".format(
                    self.config.val_size, len(tokenized_val_split)
                )
            )
            self.config.val_size = len(tokenized_val_split)

        if self.config.class_weights is not None:
            class_counts = self._compute_class_counts(tokenized_train_split)
            self.config.class_weights = self._compute_class_weights(
                class_weights=self.config.class_weights, class_counts=class_counts
            )

        types, shapes = self.feed_shape_type_def()
        if not has_targets(lambda: tokenized_train_split):
            types = types[0]
            shapes = shapes[0]

        train_dataset_unbatched = self.make_dataset_fn(
            data_fn=lambda: tokenized_train_split,
            tqdm_mode="train",
            update_hook=update_hook,
            types=types,
            shapes=shapes,
        )
        val_dataset_unbatched = self.make_dataset_fn(
            data_fn=lambda: tokenized_val_split,
            tqdm_mode="evaluate",
            types=types,
            shapes=shapes,
        )

        return {
            "train_dataset": batch_dataset(
                train_dataset_unbatched,
                batch_size=self.config.batch_size,
                shapes=shapes,
                n_epochs=self.config.n_epochs,
            ),
            "val_dataset": batch_dataset(
                val_dataset_unbatched, batch_size=self.config.batch_size, shapes=shapes
            ),
        }

    def resampling(self, Xs, Y, context=None):
        return Xs, Y, context

    @property
    def pad_idx(self):
        if self.pad_idx_ is None:
            if hasattr(self.label_encoder, "classes_"):
                classes = list(self.label_encoder.classes_)
                if self.config.pad_token in classes:
                    self.pad_idx_ = classes.index(self.config.pad_token)
                else:
                    self.pad_idx_ = None
        return self.pad_idx_

    def _format_for_encoding(self, X):
        """
        Most subclasses take in inputs as:
            List (batch) of list (docs)

        Encode_multi_input expect the following format:
            List (batch) of list (docs) of list (subseqs) of text

        This method is responsible for standardizing inputs to the above format
        """
        return [X]

    def _text_to_ids(self, Xs, pad_token=None):
        Xs = self._format_for_encoding(Xs)
        if self.config.chunk_long_sequences and len(Xs) == 1:
            # can only chunk single sequence inputs
            encoded = self.text_encoder.encode_multi_input(
                Xs,
                max_length=sys.maxsize,
                remove_repeated_whitespace=self.config.collapse_whitespace,
                include_bos_eos=self.config.include_bos_eos,
            )
            length = len(encoded.token_ids)
            field_starts_and_ends = dict()
            for field in EncodedOutput._fields:
                field_value = getattr(encoded, field)
                if field_value is not None:
                    field_starts_and_ends[field] = (field_value[0], field_value[-1])
            if self.config.chunk_context == 0 and self.config.add_eos_bos_to_chunk:
                warnings.warn("""Chunk context of 0 will not capture the start
                              and end tokens added by add_eos_bos_to_chunk""")
            for start, end, (useful_start, useful_end) in self.chunker.generate_chunks(length):
                d = dict()
                for field in EncodedOutput._fields:
                    field_value = getattr(encoded, field)
                    if field_value is not None:
                        fv = field_value[start:end]
                        if self.config.add_eos_bos_to_chunk:
                            start_token, end_token = field_starts_and_ends[field]
                            if fv[0] != start_token:
                                fv = np.concatenate(([start_token], fv))
                                # Update start and end only once
                                if field == EncodedOutput._fields[0]:
                                    useful_start += 1
                                    useful_end += 1
                            if fv[-1] != end_token:
                                fv = np.concatenate((fv, [end_token]))
                        d[field] = fv
                yield EncodedOutput(
                    useful_start=useful_start, useful_end=useful_end, input_text=Xs, **d
                )
        else:
            encoder_out = self.text_encoder.encode_multi_input(
                Xs, max_length=self.config.max_length,
                remove_repeated_whitespace=self.config.collapse_whitespace,
                include_bos_eos=self.config.include_bos_eos,
            )

            d = dict()
            for field in EncodedOutput._fields:
                field_value = getattr(encoder_out, field)
                if field_value is not None:
                    d[field] = field_value

            yield EncodedOutput(input_text=Xs, **d)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_text_encoder" in state:
            del state["_text_encoder"]
        return state


import itertools
import logging
import math
import sys
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as dataset_shuffle

from finetune.encoding.input_encoder import EncodedOutput, tokenize_context
from finetune.errors import FinetuneError
from finetune.util.imbalance import compute_class_weights
from finetune.util.input_utils import (
    Chunker,
    InputMode,
    batch_dataset,
    has_targets,
    wrap_tqdm,
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

    def _add_context_info_if_present(self, types, shapes, concrete_dims):
        if self.config.use_auxiliary_info:
            TS = tf.TensorShape
            types["context"] = tf.float32
            shapes["context"] = TS([self.config.max_length if concrete_dims else None, self.config.context_dim])
        return types, shapes

    def target_def(self, concrete_dims):
        return (tf.int32, tf.TensorShape([self.target_dim]))

    def input_spec(
        self, *, concrete_dims, include_lengths=True, batched=True, include_targets=True
    ):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([self.config.max_length if concrete_dims else None])}
        if include_lengths:
            types["length"] = tf.int32
            shapes["length"] = TS([])
        types, shapes = self._add_context_info_if_present(types, shapes, concrete_dims=concrete_dims)
        target_type, target_shape = self.target_def(concrete_dims=concrete_dims)
        if batched:
            output = (
                (types, target_type),
                tf.nest.map_structure(
                    lambda ts: tf.TensorShape(
                        [
                            self.config.batch_size if concrete_dims else None,
                            *ts.as_list(),
                        ]
                    ),
                    (shapes, target_shape),
                ),
            )
        else:
            output = ((types, target_type), (shapes, target_shape))
        if not include_targets:
            (types, _), (shapes, _) = output
            output = (types, shapes)
        return output

    def keras_input_def(self, *, concrete_dims, include_targets=False):
        input_types, input_shapes = self.input_spec(
            concrete_dims=concrete_dims, include_targets=include_targets, batched=False
        )
        return tf.nest.map_structure(
            lambda dtype, shape: tf.keras.Input(shape=shape, dtype=dtype),
            input_types,
            input_shapes,
        )

    def keras_input_signature(self, *, concrete_dims, include_targets):
        input_types, input_shapes = self.input_spec(
            concrete_dims=concrete_dims, include_targets=include_targets
        )
        return tf.nest.map_structure(
            lambda dtype, shape: tf.TensorSpec(shape=shape, dtype=dtype),
            input_types,
            input_shapes,
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
            LOGGER.debug("Outputting tokenized sequence")
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

    def make_dataset_fn(self, data_fn, tqdm_mode, shapes, types, update_hook=None):
        return tf.data.Dataset.from_generator(
            wrap_tqdm(
                gen=data_fn,
                mode=tqdm_mode,
                n_epochs=self.config.n_epochs,
                dataset_size=self.config.dataset_size,
                quiet=self.config.debugging_logs,
                update_hook=update_hook,
                current_epoch_offset=self.current_epoch_offset
                if tqdm_mode == "train"
                else 0,
                total_epoch_offset=self.total_epoch_offset
                if tqdm_mode == "train"
                else 0,
            ),
            types,
            shapes,
        )

    def get_dataset_from_generator(self, generator_fn, input_mode, update_hook=None):
        # Get from generator assumes no XLA, uses dynamic padding and batching etc.
        if input_mode != InputMode.PREDICT:
            raise ValueError(
                "From generator does not support training. Use get_dataset_from_list"
            )

        def chunked_and_tokenized_dataset():
            for d in generator_fn():
                yield from self.text_to_tokens_mask(**d)

        types, shapes = self.input_spec(
            concrete_dims=False,
            include_lengths=False,
            batched=False,
            include_targets=False,
        )
        tqdm_mode = "predict"

        raw_dataset = self.make_dataset_fn(
            data_fn=chunked_and_tokenized_dataset,
            tqdm_mode=tqdm_mode,
            update_hook=update_hook,
            types=types,
            shapes=shapes,
        )
        return {
            "predict_dataset": batch_dataset(
                raw_dataset,
                batch_size=self.config.predict_batch_size,
                max_length=self.config.max_length,
                table_batching=False,  # We cannot use table batching here because the order of the outputs is impacted.
                shapes=shapes,
                drop_remainder=False,
            )
        }

    def get_dataset_from_list(self, data_list, input_mode, update_hook=None):
        # From list assumes we will be compiling with XLA, we used fixed batch_size batches and max_length sequences.
        # We also drop up to batch_size - 1 items from the final batch of the final epoch. if that is necessary to maintain even batch sizes.
        if input_mode != InputMode.TRAIN:
            raise ValueError(
                "From list does not support prediction. Use get_dataset_from_generator"
            )

        data_list = list(data_list)
        self._post_data_initialization(data_list)

        train_split = dataset_shuffle(data_list, random_state=self.config.seed)

        tokenized_train_split = list(
            itertools.chain.from_iterable(
                self.text_to_tokens_mask(**d) for d in train_split
            )
        )

        self.config.dataset_size = len(tokenized_train_split)

        if self.config.class_weights is not None:
            class_counts = self._compute_class_counts(tokenized_train_split)
            self.config.class_weights = self._compute_class_weights(
                class_weights=self.config.class_weights, class_counts=class_counts
            )

        types, shapes = self.input_spec(
            concrete_dims=False, include_lengths=False, batched=False
        )
        _, concrete_shapes = self.input_spec(
            concrete_dims=True, include_lengths=False, batched=False
        )

        if self.config.min_steps is not None:
            self.config.n_epochs = max(
                self.config.n_epochs,
                math.ceil(self.config.min_steps / self.config.dataset_size),
            )

        train_dataset_unbatched = self.make_dataset_fn(
            data_fn=lambda: tokenized_train_split,
            tqdm_mode="train",
            update_hook=update_hook,
            types=types,
            shapes=shapes,
        )
        batched_train_dataset = batch_dataset(
            train_dataset_unbatched,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            shapes=concrete_shapes,
            n_epochs=self.config.n_epochs,
            shuffle=self.config.reshuffle_chunks,
            table_batching=self.config.table_batching,
            random_seed=self.config.seed,
            drop_remainder=True,
        )
        return {
            "train_dataset": batched_train_dataset,
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
                if field_value is not None and len(field_value):
                    field_starts_and_ends[field] = (field_value[0], field_value[-1])
            if self.config.chunk_context == 0 and self.config.add_eos_bos_to_chunk:
                warnings.warn(
                    """Chunk context of 0 will not capture the start
                              and end tokens added by add_eos_bos_to_chunk"""
                )
            for start, end, (useful_start, useful_end) in self.chunker.generate_chunks(
                length
            ):
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
                LOGGER.debug("Yielding tokenized sequence")
                yield EncodedOutput(
                    useful_start=useful_start, useful_end=useful_end, input_text=Xs, **d
                )
        else:
            encoder_out = self.text_encoder.encode_multi_input(
                Xs,
                max_length=self.config.max_length,
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

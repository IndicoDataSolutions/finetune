import os
import gc
import random
import weakref
import atexit
import warnings
import itertools
import math
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import tempfile
import time
import sys
from contextlib import contextmanager
import pathlib
import logging

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.compat.v1 import logging as tf_logging

from sklearn.model_selection import train_test_split
import joblib

from finetune.util import list_transpose
from finetune.encoding.input_encoder import EncodedOutput
from finetune.config import all_gpus, assert_valid_config, get_default_config
from finetune.saver import Saver, InitializeHook
from finetune.errors import FinetuneError
from finetune.model import get_model_fn, PredictMode
from finetune.util.download import download_data_if_required
from finetune.util.shapes import shape_list
from finetune.util.timing import ProgressBar
from finetune.util.in_memory_finetune import make_in_memory_finetune_hooks
from finetune.util.indico_estimator import IndicoEstimator
from finetune.util.gpu_info import gpu_info

from finetune.base_models.bert.model import _BaseBert
from finetune.base_models import GPTModel, GPTModelSmall
from finetune.input_pipeline import InputMode

LOGGER = logging.getLogger("finetune")


class SSLPipeline(BasePipeline):
    def __init__(self, config, multi_label):
        super(SequencePipeline, self).__init__(config)

    def zip_list_to_dict(self, X, Y=None, context=None):
        if Y is not None:
            Y = list(Y)
            if len(X) != len(Y):
                raise FinetuneError("the length of your labels does not match the length of your text")
        if context is not None:
            context = list(context)
            if len(X) != len(context):
                raise FinetuneError("the length of your context does not match the length of your text")
        x_out = []
        for i, x in enumerate(X):
            sample = {"X": x}
            if Y is not None:
                sample["Y"] = Y[i]
            if context is not None:
                sample["context"] = context[i]
            x_out.append(sample)
        return out

    def  text_to_tokens_mask(self, X, Y=None, context=None):
        pad_token = [self.config.pad_token] if self.multi_label else self.config.pad_token
        out_gen = self._text_to_ids(X, pad_token=pad_token)
        for out in out_gen:
            feats = {"tokens": out.token_ids}
            if context is not None:
                tokenized_context = tokenize_context(context, out, self.config)
                feats['context'] = tokenized_context
            if Y is None:
                yield feats
            if Y is not None:
                min_starts = min(out.token_starts)
                max_ends = max(out.token_ends)
                filtered_labels = [
                    lab for lab in Y if lab["end"] >= min_starts and lab["start"] <= max_ends
                ]
                if self.config.filter_empty_examples and len(filtered_labels) == 0:
                    continue
                yield feats, self.label_encoder.transform(out, filtered_labels)

    def _compute_class_counts(self, encoded_dataset):
        counter = Counter()
        for doc, target_arr in encoded_dataset:
            target_arr = np.asarray(target_arr)
            decoded_targets = self.label_encoder.inverse_transform(target_arr)
            if self.multi_label:
                for label in decoded_targets:
                    counter.update(label)
            else:
                counter.update(decoded_targets)
        return counter

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = (
            [None, self.label_encoder.target_dim]
            if self.multi_label
            else [None]
        )
        return (
            (types, tf.float32,),
            (shapes, TS(target_shape),),
        )

    def make_dataset_fn(self, data_fn, tqdm_mode, shapes, types, update_hook=None, skip_val=False):
        def dataset_fn():
            return Dataset.from_generator(
                wrap_tqdm(
                    gen=data_fn,
                    mode=tqdm_mode,
                    n_epochs=self.config.n_epochs,
                    val_size=self.config.val_size,
                    dataset_size=self.config.dataset_size,
                    skip_val=skip_val,
                    silent=self.config.debugging_logs,
                    update_hook=update_hook
                ),
                types,
                shapes
            )
        return dataset_fn

    def get_dataset_from_generator(self, generator_fn, input_mode,
                                   update_hook=None, u_generator_fn=None):
        def chunked_and_tokenized_dataset(gen):
            for d in gen():
                yield from self.text_to_tokens_mask(**d)

        types, shapes = self.feed_shape_type_def()
        
        if input_mode == InputMode.PREDICT:
            tqdm_mode = "predict"
        else:
            tqdm_mode = "train"

        if input_mode == InputMode.PREDICT or not has_targets(generator_fn):
            x_types, x_shapes = types[0], shapes[0]
        else
            x_types, x_shapes = types, shapes
       
        x_data_fn = lambda: chunked_and_tokenized_dataset(generator_fn)
        x_raw_dataset = self.make_dataset_fn(
            data_fn=x_data_fn,
            tqdm_mode=tqdm_mode,
            update_hook=update_hook,
            types=x_types,
            shapes=x_shapes,
            skip_val=input_mode == InputMode.TRAIN
        )

        
        if input_mode == InputMode.PREDICT:
            if u_generator_fn:
                warnings.warn("""U is ignored in predict mode - pass all data
                              as X for predictions""")
            return {
                "predict_dataset": batch_dataset(
                    x_raw_dataset,
                    batch_size=self.config.predict_batch_size,
                    shapes=shapes,
                )
            }
        
        if self.config.chunk_long_sequences:
            LOGGER.warning("The dataset size is not adjusted for chunk long sequences when training from a generator")

        if self.config.dataset_size is None:
            raise FinetuneError("If you are using a callable as input you must provide config.dataset_size")

        if self.config.class_weights is not None or self.config.oversample:
            raise FinetuneError("Cannot use class weights or resampling in generator mode")

        self.config.val_size, self.config.val_interval = validation_settings(
            dataset_size=self.config.dataset_size,
            batch_size=self.config.batch_size,
            val_size=self.config.val_size,
            val_interval=self.config.val_interval,
            keep_best_model=self.config.keep_best_model
        )

        self.config.dataset_size -= self.config.val_size

        val_dataset = (
            lambda: x_raw_dataset()
            .shuffle(
                self.config.shuffle_buffer_size,
                seed=self.config.seed,
                reshuffle_each_iteration=False,
            )
            .take(self.config.val_size)
        )
        x_train_dataset = (
            lambda: x_raw_dataset()
            .shuffle(
                self.config.shuffle_buffer_size,
                seed=self.config.seed,
                reshuffle_each_iteration=False,
            )
            .skip(self.config.val_size)
        )
        x_batch_dataset = batch_dataset(
                train_dataset,
                batch_size=self.config.batch_size,
                shapes=shapes,
                n_epochs=self.config.n_epochs
        )

        if u_generator_fn:
            u_data_fn = lambda: chunked_and_tokenized_dataset(u_generator_fn)
            u_types, u_shapes = types[0], shapes[0]
            u_raw_dataset = self.make_dataset_fn(
                data_fn=u_data_fn,
                tqdm_mode=tqdm_mode,
                update_hook=update_hook,
                types=u_types,
                shapes=u_shapes,
                skip_val=input_mode == InputMode.TRAIN
            )
            u_train_dataset = (
                lambda: u_raw_dataset()
                .shuffle(
                    self.config.shuffle_buffer_size,
                    seed=self.config.seed,
                    reshuffle_each_iteration=False,
                )
            )
            u_batch_dataset = batch_dataset(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shapes=shapes,
                    n_epochs=self.config.n_epochs
            )
            train_gen = lambda: combine_datasets(x_batch_dataset,
                                                 u_batch_dataset)
        else:
            train_gen = x_batch_dataset

        return {
            "train_dataset": train_gen,
            "val_dataset": batch_dataset(
                val_dataset,
                batch_size=self.config.batch_size,
                shapes=shapes
            )
        }


    def get_dataset_from_list(self, data_list, input_mode, update_hook=None, u_data_list=None):
        assert input_mode == InputMode.TRAIN, "use the generator path for prediction"
        
        x_data_list = list(data_list)
        self._post_data_initialization(x_data_list)
            
        self.config.val_size, self.config.val_interval = validation_settings(
            dataset_size=len(x_data_list),
            batch_size=self.config.batch_size,
            val_size=self.config.val_size,
            val_interval=self.config.val_interval,
            keep_best_model=self.config.keep_best_model
        )

        if self.config.val_size > 0 and self.config.val_set is None:
            train_split, val_split = train_test_split(x_data_list, test_size=self.config.val_size, random_state=self.config.seed)
        else:
            train_split = dataset_shuffle(x_data_list, random_state=self.config.seed)
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
        
        if self.config.class_weights is not None:
            class_counts = self._compute_class_counts(tokenized_train_split)
            self.config.class_weights = self._compute_class_weights(
                class_weights=self.config.class_weights,
                class_counts=class_counts
            )
            
        types, shapes = self.feed_shape_type_def()
        if not has_targets(lambda: tokenized_train_split):
            x_types, x_shapes = types[0], shapes[0]
        else
            x_types, x_shapes = types, shapes

        train_split_unbatched = self.make_dataset_fn(
            data_fn=lambda: tokenized_train_split,
            tqdm_mode="train",
            update_hook=update_hook,
            types=x_types,
            shapes=x_shapes
        )
        val_dataset_unbatched = self.make_dataset_fn(
            data_fn=lambda: tokenized_val_split,
            tqdm_mode="evaluate",
            types=types,
            shapes=shapes
        )

        train_split_batched = batch_dataset(
            train_split_unbatched,
            batch_size=self.config.batch_size,
            shapes=x_shapes,
            n_epochs=self.config.n_epochs
        )

        if u_data_list:
            u_data_list = list(u_data_list)
            u_train = dataset_shuffle(u_data_list, random_state=self.config.seed)
            tokenized_u_train = list(
                itertools.chain.from_iterable(
                    self.text_to_tokens_mask(**d) for d in u_train
                )
            )
            u_types, u_shapes = types[0], shapes[0]
            u_train_unbatched = self.make_dataset_fn(
                data_fn=lambda: tokenized_u_train,
                tqdm_mode="train",
                update_hook=update_hook,
                types=u_types,
                shapes=u_shapes
            )
            u_train_batched = batch_dataset(
                u_train_unbatched,
                batch_size=self.config.batch_size,
                shapes=u_shapes,
                n_epochs=self.config.n_epochs
            )
            train_gen = lambda: combine_datasets(train_split_batched,
                                                 u_train_batch)
        else:
            train_gen = train_split_batched

        return {
            "train_dataset": train_gen,
            "val_dataset": batch_dataset(
                val_dataset_unbatched,
                batch_size=self.config.batch_size,
                shapes=shapes
            )
        }

    def combine_datasets(x_dataset, u_dataset):
        for X, U in zip(x_dataset, u_dataset):
            combined = {
                **X,
                "u_tokens": U["tokens"]
            }
            if "context" in U:
                combined["u_context"] = U["context"]
            return combined

    def _target_encoder(self):
        return SequenceLabelingEncoder(pad_token=self.config.pad_token)

class SSLLabeler(BaseModel):
    defaults = dict()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return SSLPipeline(config=self.config)

    def _initialize(self):
        return super()._initialize()

    def predict(self, X, per_token=False, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param per_token: If True, return raw probabilities and labels on a per token basis
        :returns: list of class labels.
        """
        return super().predict(X, per_token=per_token, context=context, **kwargs)

    def predict_proba(self, X, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :returns: list of class labels.
        """
        return self.predict(X, context=context, **kwargs)

    def _predict_op(self, logits, **kwargs):
        pass

    def _predict_proba_op(self, logits, **kwargs):
        pass

    def featurize(self, X, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X, **kwargs)

    def _target_model(
        self, *, config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs
    ):
        return sequence_labeler(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            multilabel=config.multi_label_sequences,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            **kwargs
        )

    def finetune(self, Xs, Us=None, Y=None, context=None, update_hook=None):
        if callable(Xs):
            assert (not Us or callable(Us)), "If X is a generator, U must also be a generator"
            datasets = self.input_pipeline.get_dataset_from_generator(
                Xs, input_mode=InputMode.TRAIN,
                update_hook=update_hook, u_generator_fn=Us
            )
        else:
            assert (not Us or not callable(Us)), "If X is a list, U must also be a list"
            x_list = self.input_pipeline.zip_list_to_dict(X=Xs, Y=Y, context=context)
            u_list = self.input_pipeline.zip_list_to_dict(X=Us, Y=Y, context=context)
            datasets = self.input_pipeline.get_dataset_from_list(
                x_list, input_mode=InputMode.TRAIN,
                update_hook=update_hook, u_data_list=u_list
            )
                
        if self.config.keep_best_model:
            if self.config.val_size <= 10:
                tf.compat.v1.logging.warning(
                    "Early stopping / keeping best model with a validation size of {} is likely to case undesired results".format(
                        self.config.val_size
                    )
                )

        force_build_lm = Y is None
        estimator, hooks = self.get_estimator(force_build_lm=force_build_lm)
        train_hooks = hooks.copy()

        steps_per_epoch = self._n_steps(
            n_examples=self.input_pipeline.dataset_size,
            batch_size=self.config.batch_size,
            n_gpus=max(1, len(self.resolved_gpus)),
        )
        num_steps = steps_per_epoch * self.config.n_epochs

        if self.config.val_size > 0:
            # Validation with all other tasks.
            train_hooks.append(
                tf.estimator.experimental.InMemoryEvaluatorHook(
                    estimator,
                    datasets["val_dataset"],
                    every_n_iter=self.config.val_interval,
                    steps=math.ceil(self.config.val_size / self.config.batch_size),
                )
            )
            early_stopping_interval = self.config.val_interval
        else:
            early_stopping_interval = sys.maxsize

        train_hooks.append(
            self.saver.get_saver_hook(
                estimator=estimator,
                keep_best_model=self.config.keep_best_model,
                steps_per_epoch=steps_per_epoch,
                early_stopping_steps=self.config.early_stopping_steps,
                eval_frequency=early_stopping_interval,
                cache_weights_to_file=self.config.cache_weights_to_file
            )
        )

        if self.config.in_memory_finetune is not None:
            train_hooks.extend(make_in_memory_finetune_hooks(self, estimator))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_input_fn_skipped = lambda: datasets["train_dataset"]().skip(self.saver.get_initial_step() * max(len(self.resolved_gpus), 1))
            estimator.train(train_input_fn_skipped, hooks=train_hooks, steps=num_steps)
        
        self._trained = True

    def _predict(self, zipped_data, **kwargs):
        raise NotImplemented()

    def _predict_proba(self, zipped_data, **kwargs):
        pass

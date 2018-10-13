import itertools
import logging
import sys
import math
from abc import ABCMeta, abstractmethod

import random
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

from finetune.errors import FinetuneError
from finetune.config import PAD_TOKEN
from finetune.encoding import TextEncoder, ArrayEncodedOutput, EncodedOutput
from finetune.imbalance import compute_class_weights

ENCODER = TextEncoder()
LOGGER = logging.getLogger('finetune')


class ProgressBar(object):
  
    def __init__(self, input_pipeline, batch_size, n_examples=None, val_size=0, mode='train'):
        if mode not in ('train', 'predict'):
            raise FinetuneError("Invalid value for `TQDM` mode: {}".format(mode))
        self.mode = mode
        self.input_pipeline = input_pipeline
        self.iterations = 0
        self.n_epochs = self.input_pipeline.config.n_epochs
        self.progress_bar = None
        self.batch_size = batch_size
        self.val_size = val_size
        self.n_examples = n_examples 

    def epoch_descr(self, current_epoch):
        return "Epoch {}/{}".format(current_epoch, self.n_epochs)
    
    def write_description(self, current_epoch):
        if self.mode == 'train':
            self.progress_bar.set_description(self.epoch_descr(current_epoch))
        else:
            self.progress_bar.set_description("Inference")
    
    def tf_log_progress(self, *args):
        with tf.control_dependencies([tf.py_func(self.log_progress, (), ())]):
            return args
        
    def log_progress(self):
        self.iterations += 1

        if self.progress_bar is None:
            self.n_examples = self.n_examples or self.input_pipeline.config.dataset_size
            self.n_examples -= self.val_size
            self.n_batches = math.ceil(self.n_examples / self.batch_size)
            self.progress_bar = tqdm.tqdm(total=self.n_batches)

        current_epoch = self.iterations // self.n_batches + 1
        current_example = self.iterations % self.n_batches

        if current_example == 0 and current_epoch != 1:
            current_epoch -= 1
            current_example = self.n_batches

        if self.progress_bar is not None:
            self.write_description(current_epoch)

            self.progress_bar.n = current_example
            self.progress_bar.refresh()

    def __del__(self):
        # ensure flush to stdout before deletion
        if self.progress_bar is not None:
            self.progress_bar.refresh()


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        self.label_encoder = None
        self.target_dim = None
        self.pad_idx_ = None
        self.rebuild = False

    @abstractmethod
    def _target_encoder(self):
        # Overridden by subclass to produce the right target encoding for a given target model.
        raise NotImplementedError

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        return ({"tokens": tf.int32, "mask": tf.float32}, tf.float32), (
            {"tokens": TS([self.config.max_length, 2]), "mask": TS([self.config.max_length])}, TS([self.target_dim]))

    def _array_format(self, encoded_output, pad_token=PAD_TOKEN):
        """
        Returns numpy array of token idxs and corresponding mask
        Returned `x` array contains two channels:
            0: byte-pair encoding embedding
            1: positional embedding
        """
        seq_length = len(encoded_output.token_ids)
        x = np.zeros((self.config.max_length, 2), dtype=np.int32)
        mask = np.zeros((self.config.max_length), dtype=np.float32)

        if encoded_output.labels is not None:
            labels_arr = np.empty((self.config.max_length), dtype='object')
            labels_arr.fill(pad_token)
        else:
            labels_arr = None

        # BPE embedding
        x[:seq_length, 0] = encoded_output.token_ids
        # masking: value of 1 means "consider this in cross-entropy LM loss"
        mask[1:seq_length] = 1
        if encoded_output.labels:
            labels_arr[:seq_length] = encoded_output.labels
        # positional_embeddings
        x[:, 1] = np.arange(ENCODER.vocab_size, ENCODER.vocab_size + self.config.max_length)

        return ArrayEncodedOutput(
            token_ids=x,
            tokens=encoded_output.tokens,
            labels=labels_arr,
            char_locs=encoded_output.char_locs,
            mask=mask,
        )

    def text_to_tokens_mask(self, X, Y=None):
        out_gen = self._text_to_ids(X)
        for out in out_gen:
            feats = {"tokens": out.token_ids, "mask": out.mask}
            if Y is None:
                yield feats
            else:
                yield feats, self.label_encoder.transform([Y])[0]

    def _post_data_initialization(self, Y):
        self.label_encoder = self._target_encoder()
        if not callable(Y):
            Y_fit = Y
            self.label_encoder.fit(Y)
        else:
            Y_fit = list(itertools.islice(Y(), 100))  # TODO find a more principled way to do this?
            self.label_encoder.fit(Y_fit)

        target_dim = self.label_encoder.target_dim
        self.lm_loss_coef = self.config.lm_loss_coef if target_dim is not None else 1.0
        self.target_dim = target_dim

        if Y_fit is not None:
            self.config.class_weights = compute_class_weights(class_weights=self.config.class_weights, Y=Y_fit)

    def _dataset_with_targets(self, Xs, Y, mode='train'):
        if not callable(Xs) and not callable(Y):
            dataset = lambda: zip(Xs, Y)
        elif callable(Xs) and callable(Y):
            dataset = lambda: zip(Xs(), Y())  # encode one sample at a time.
        else:
            raise ValueError("Either neither or both of Xs and Y should be callable, not a mixture")

        dataset_encoded = lambda: itertools.chain.from_iterable(
            map(lambda xy: self.text_to_tokens_mask(*xy), dataset()))
        shape_def = self.feed_shape_type_def()
        if not callable(Y) and self.config.chunk_long_sequences:
            dataset_encoded_list = list(dataset_encoded())  # come up with a more principled way to do this.
            dataset_encoded = lambda: dataset_encoded_list
            self.config.dataset_size = len(dataset_encoded_list)
        return Dataset.from_generator(dataset_encoded, *shape_def)

    def _dataset_without_targets(self, Xs):
        if not callable(Xs):
            Xs_fn = lambda: Xs
            self._n_examples = len(Xs)
        else:
            Xs_fn = Xs
            try:
                self._n_examples = len(Xs)
            except TypeError:
                pass
        
        dataset_encoded = lambda: itertools.chain.from_iterable(map(self.text_to_tokens_mask, Xs_fn()))
        types, shapes = self.feed_shape_type_def()
        return Dataset.from_generator(dataset_encoded, types[0], shapes[0])  # 0s cut out the targets

    def validation_settings(self, n_examples, batch_size):
        """
        Auto-select reasonable validation settings
        """
        if self.config.val_size is not None and self.config.val_interval is not None:
            return self.config.val_size, self.config.val_interval

        # Auto-select reasonable validation size
        if self.config.val_size is None:
            if n_examples < 50:
                val_size = 0
            else:
                val_size = max(5, int(0.05 * n_examples))
                val_size = min(100, val_size)
        else:
            val_size = self.config.val_size

        # Auto-select reasonable validation interval
        if self.config.val_interval is None:
            # sys.maxsize corresponds to never running validation
            # and is used when val_size is set to 0
            val_interval = 4 * int(math.ceil(val_size / batch_size)) or sys.maxsize
        else:
            val_interval = self.config.val_interval

        return int(val_size), int(val_interval)

    def get_train_input_fns(self, Xs, Y=None, batch_size=None):
        batch_size = batch_size or self.config.batch_size

        shuffle_buffer_size = self.config.shuffle_buffer_size
        prefetch_buffer = 2  # breaks the pipeline to allow concurrency

        if callable(Xs):
            try:
                self.config.dataset_size = len(Xs())
            except TypeError:
                if self.config.dataset_size is None:
                    raise FinetuneError(
                        "Generator input function does not have a length and no `config.dataset_size` is specified. "
                        "You must set `config.dataset_size` explicitly."
                    )
        else:
            self.config.dataset_size = len(Xs)

        val_size, val_interval = self.validation_settings(
            n_examples=len(Xs) if not callable(Xs) else self.config.dataset_size,
            batch_size=batch_size or self.config.batch_size
        )
        self.config.val_size = val_size

        if Y is not None:
            self._post_data_initialization(Y)
            dataset = lambda: self._dataset_with_targets(Xs, Y)
        else:
            dataset = lambda: self._dataset_without_targets(Xs)

        train_tqdm = ProgressBar(self, batch_size=batch_size, val_size=val_size)
        val_dataset = lambda: dataset().take(
            val_size).shuffle(shuffle_buffer_size, seed=self.config.seed).batch(batch_size, drop_remainder=False).prefetch(prefetch_buffer)
        train_dataset = lambda: dataset().skip(
            val_size).shuffle(shuffle_buffer_size, seed=self.config.seed).batch(batch_size, drop_remainder=False).repeat(self.config.n_epochs).map(
                train_tqdm.tf_log_progress
            ).prefetch(prefetch_buffer)
        return val_dataset, train_dataset, val_size, val_interval

    def get_predict_input_fn(self, Xs, batch_size=None):
        batch_size = batch_size or self.config.batch_size
        prefetch_buffer = 2  # breaks the pipeline to allow concurrency
        tf_dataset = lambda: self._dataset_without_targets(Xs)
        return lambda: tf_dataset().batch(batch_size).prefetch(prefetch_buffer)

    @property
    def pad_idx(self):
        if self.pad_idx_ is None:
            self.pad_idx_ = list(self.label_encoder.classes_).index(self.config.pad_token)
        return self.pad_idx_

    def _format_for_encoding(self, X):
        """
        Most subclasses take in inputs as:
            List (batch) of list (docs)

        Encode_multi_input expect the following format:
            List (batch) of list (docs) of list (subseqs) of text

        This method is responsible for standardizing inputs to the above format
        """
        return [[X]]

    def _text_to_ids(self, Xs, Y=None, pad_token=PAD_TOKEN):
        Xs = self._format_for_encoding(Xs)
        if self.config.chunk_long_sequences and len(Xs) == 1:
            # can only chunk single sequence inputs
            chunk_size = self.config.max_length - 2
            step_size = chunk_size // 3
            encoded = ENCODER.encode_multi_input(
                Xs,
                Y=Y,
                max_length=sys.maxsize,
                pad_token=pad_token
            )
            length = len(encoded.token_ids)
            starts = list(range(0, length, step_size))
            for start in starts:
                d = dict()
                end = start + chunk_size
                for field in EncodedOutput._fields:
                    field_value = getattr(encoded, field)
                    if field_value is not None:
                        d[field] = field_value[start:end]
                yield self._array_format(EncodedOutput(**d), pad_token=pad_token)
        else:
            encoder_out = ENCODER.encode_multi_input(
                Xs,
                Y=Y,
                max_length=self.config.max_length,
                pad_token=pad_token
            )

            yield self._array_format(encoder_out, pad_token=pad_token)
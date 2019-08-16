import itertools
import logging
import sys
import math
import os
from collections import Counter

from abc import ABCMeta, abstractmethod

import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import finetune
from finetune.errors import FinetuneError
from finetune.encoding.input_encoder import ArrayEncodedOutput, EncodedOutput
from finetune.util.imbalance import compute_class_weights

LOGGER = logging.getLogger("finetune")


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        self.text_encoder = self.config.base_model.get_encoder()
        self.context_dim = None
        self.label_encoder = None
        self.target_dim = None
        self.pad_idx_ = None
        self.rebuild = False
        self.epoch = 0

    @property
    def dataset_size(self):
        return self.config.dataset_size

    @abstractmethod
    def _target_encoder(self):
        # Overridden by subclass to produce the right target encoding for a given target model.
        raise NotImplementedError

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        if self.config.use_auxiliary_info:
            return (
                (
                    {"tokens": tf.int32, "mask": tf.float32, "context": tf.float32},
                    tf.float32,
                ),
                (
                    {
                        "tokens": TS([self.config.max_length, 2]),
                        "mask": TS([self.config.max_length]),
                        "context": TS([self.config.max_length, self.context_dim]),
                    },
                    TS([self.target_dim]),
                ),
            )
        else:
            return (
                ({"tokens": tf.int32, "mask": tf.float32}, tf.float32),
                (
                    {
                        "tokens": TS([self.config.max_length, 2]),
                        "mask": TS([self.config.max_length]),
                    },
                    TS([self.target_dim]),
                ),
            )

    def _array_format(self, encoded_output, pad_token=None):
        """
        Returns numpy array of token idxs and corresponding mask
        Returned `x` array contains two channels:
            0: byte-pair encoding embedding
            1: positional embedding
        """
        seq_length = len(encoded_output.token_ids)
        x = np.zeros((self.config.max_length, 2), dtype=np.int32)
        if self.config.base_model.__name__ == "RoBERTa":
            x += 1
        mask = np.zeros((self.config.max_length), dtype=np.float32)

        if encoded_output.labels is not None:
            labels_arr = np.empty((self.config.max_length), dtype="object")
            labels_arr.fill((pad_token or self.config.pad_token))
        else:
            labels_arr = None

        if encoded_output.context is not None:
            context_arr = np.zeros(
                (self.config.max_length, self.context_dim), dtype=np.float32
            )
        else:
            context_arr = None

        # BPE embedding
        x[:seq_length, 0] = encoded_output.token_ids
        # masking: value of 1 means "consider this in cross-entropy LM loss"
        mask[1:seq_length] = 1
        if encoded_output.labels:
            labels_arr[:seq_length] = encoded_output.labels
        if encoded_output.context is not None:
            if len(np.shape(encoded_output.context)) in (2, 3):
                context_arr[:seq_length][:] = np.squeeze(encoded_output.context)
            else:
                raise FinetuneError("Incorrect context rank.")

        # positional_embeddings
        x[:, 1] = np.arange(
            self.text_encoder.vocab_size,
            self.text_encoder.vocab_size + self.config.max_length,
        )

        # roberta uses different positional embedding structure
        if self.config.base_model.__name__ == "RoBERTa":
            mask = np.zeros((self.config.max_length), dtype=np.float32)
            mask[0:seq_length] = 1
            positions = np.cumsum(mask, dtype=np.int32)
            positions += 1  # add padding idx because RoBERTa's pos embeds depend on it
            positions += (
                self.text_encoder.vocab_size + 1
            )  # + 1 to include unused mask token in embedding layer
            x[:, 1] = positions

        output = ArrayEncodedOutput(
            token_ids=x,
            tokens=encoded_output.tokens,
            labels=labels_arr,
            char_locs=encoded_output.char_locs,
            mask=mask,
            context=context_arr,
        )
        return output

    def text_to_tokens_mask(self, X, Y=None, context=None):
        if context is None and self.config.use_auxiliary_info:
            context = X[1]
            X = X[0]

        out_gen = self._text_to_ids(X, pad_token=self.config.pad_token, context=context)

        for out in out_gen:
            if self.config.use_auxiliary_info:
                feats = {
                    "tokens": out.token_ids,
                    "mask": out.mask,
                    "context": out.context,
                }
            else:
                feats = {"tokens": out.token_ids, "mask": out.mask}
            if Y is None:
                yield feats
            else:
                yield feats, self.label_encoder.transform([Y])[0]

    def _post_data_initialization(self, Y=None, context=None):
        if Y is not None:
            self.label_encoder = self._target_encoder()
            if not callable(Y):
                Y_fit = Y
                self.label_encoder.fit(Y)
            else:
                Y_fit = list(itertools.islice(Y(), 10000))
                self.label_encoder.fit(Y_fit)
            self.config.pad_idx = self.pad_idx

            target_dim = self.label_encoder.target_dim
            self.lm_loss_coef = (
                self.config.lm_loss_coef if target_dim is not None else 1.0
            )
            self.target_dim = target_dim

        if context:
            if not callable(Y):
                characteristics = context
            else:
                characteristics = itertools.islice(context(), 10000)
            _ = self._context_to_vector(
                characteristics
            )  # to fit the auxiliary information label encoders

    def _context_to_vector(self, context):
        """
        Takes list of context dictionaries and turns them into lists (per sample) of lists (per token) of context vectors
        """
        num_samples = len(context)
        vector_list = []
        ignore = ["start", "end", "label", "token"]

        if not hasattr(
            self, "label_encoders"
        ):  # we will fit necessary label encoders here to know dimensionality of input vector before entering featurizer
            valid_samples = [
                dict_list for dict_list in context if dict_list != self.config.pad_token
            ]  # one list of dicts per sample of text
            pads_removed = [
                dictionary
                for dict_list in valid_samples
                for dictionary in dict_list
                if dictionary != self.config.pad_token
            ]  # concat each dictionary (per token) into a list of dict
            keys = [
                k
                for k in pads_removed[0].keys()
                if k not in ignore and k in self.default_context.keys()
            ]
            characteristics = {
                k: [dictionary[k] for dictionary in pads_removed] for k in keys
            }
            self.label_encoders = {
                k: LabelBinarizer()
                for k in characteristics.keys()
                if all(
                    k != self.config.pad_token
                    and not (
                        type(data) == int or (type(data) == float and not pd.isna(data))
                    )
                    and k not in ignore
                    for data in characteristics[k]
                )
            }  # Excludes features that are continuous, like position and font size, since they do not have categorical binary encodings

            for label, encoder in self.label_encoders.items():  # fit encoders
                without_nans = [str(x) for x in characteristics[label] if not pd.isna(x)]
                encoder.fit(without_nans)

            self.context_labels = sorted(
                [label for label in characteristics.keys() if label != "label"]
            )  # sort for consistent ordering between runs
            continuous_labels = [
                label
                for label in self.context_labels
                if label not in self.label_encoders.keys() and label not in ignore
            ]

            self.label_stats = {}
            for label in continuous_labels:
                stats = {
                    "mean": np.mean(characteristics[label]),
                    "std": np.std(characteristics[label]),
                }
                self.label_stats[label] = stats

            self.context_dim = len(
                continuous_labels
            )  # Sum over features to determine dimensionality of each feature vector.

            for (
                encoder
            ) in (
                self.label_encoders.values()
            ):  # since categorical labels use LabelBinarizer, they use varying dimensions each for each one-hot-encoding, while numerical features only use 1 dimension, so we sum over all for the total dimension.
                self.context_dim += (
                    1 if len(encoder.classes_) <= 2 else len(encoder.classes_)
                )  # Binary encodings with only two classes are given with one bit. Encodings with n > 2 classes are given with n bits. (Thanks sklearn)
            return True

        for sample in context:
            # See which tokens have padded labels/context vectors, and thus have a zero vector for features
            padded_indices = []
            tokens_with_context = []
            for idx in range(len(sample)):
                if type(sample[idx]) == str and sample[idx] == self.config.pad_token:
                    padded_indices.append(idx)
                else:
                    if not set(self.default_context.keys()).issubset(set(sample[idx].keys())):
                        missing = str(
                            set(self.default_context.keys()).difference(set(sample[idx].keys()))
                        )

                        raise FinetuneError(
                            "Token '{}' does not contain field(s) {} as described in default".format(
                                sample[idx]["token"], missing
                            )
                        )
                    tokens_with_context.append(sample[idx])
            characteristics = {
                k: [dictionary[k] for dictionary in tokens_with_context]
                for k in self.default_context.keys()
            }

            # make sure all features cover the same number of tokens, and calculate total num tokens
            num_tokens = None
            for label in self.context_labels:
                new_length = (
                    len(characteristics[label])
                    if type(characteristics[label] == list)
                    else 1
                )
                if num_tokens is not None and num_tokens != new_length:
                    raise FinetuneError("Incorrect label shapes.")
                num_tokens = new_length

            vector = np.zeros(
                (num_tokens + len(padded_indices), self.context_dim), dtype=np.float32
            )  # Feature vector for one document. Add 2 for the special tokens at beginning/end
            current_index = 0

            # Loop through each feature and add each to new index of the feature vector
            for label in self.context_labels:
                if label == self.config.pad_token or label in ignore:
                    continue
                data = characteristics[label]

                # Binary encoded features have different dimensionality as simple floats/ints, so must be handled differently.
                if label in self.label_encoders.keys():
                    without_nans = [x for x in data if not pd.isna(x)]
                    data = (
                        np.asarray(self.label_encoders[label].transform(without_nans))
                        - 0.5
                    )  # this removes nans from padded tokens, but now the list is too short. The 'num_backward' variable will track this offset to ensure indices are correctly tracked.
                    data_dim = len(self.label_encoders[label].classes_)
                    if data_dim == 2:  # since binary classes default to column vector
                        data_dim = 1
                else:  # need to normalize our float/int inputs
                    data = [str(x) for x in data if not pd.isna(x)]
                    stats = self.label_stats[label]
                    data = (data - stats["mean"]) / stats["std"]
                    data_dim = 1

                # loop through indices and fill with correct data
                tokens_added = 0
                for sample_idx in range(len(sample)):
                    if (
                        sample_idx in padded_indices
                    ):  # there is no data, simply a pad from the encoder, so fill out with zero vector
                        vector[sample_idx][:] = 0
                        continue
                    for label_dimension in range(data_dim):
                        vector[sample_idx][current_index + label_dimension] = (
                            data[tokens_added]
                            if data_dim == 1
                            else data[tokens_added][label_dimension]
                        )
                    tokens_added += 1
                current_index += 1
            vector_list.append(vector)
        return vector_list

    def _compute_class_counts(self, encoded_dataset):
        target_arrs = np.asarray([target_arr for doc, target_arr in encoded_dataset])
        return Counter(self.label_encoder.inverse_transform(target_arrs))

    def _dataset_with_targets(self, Xs, Y, train, context=None):
        if not callable(Xs) and not callable(Y):
            if self.config.use_auxiliary_info:
                dataset = lambda: zip(
                    Xs, Y, context
                )  # Do not need to check if context is callable - it is turned in along with Xs, and thus must have the same form
            else:
                dataset = lambda: zip(Xs, Y)
        elif callable(Xs) and callable(Y):
            if self.config.use_auxiliary_info:
                dataset = lambda: zip(
                    Xs(), Y(), context()
                )  # encode one sample at a time.
            else:
                dataset = lambda: zip(Xs(), Y())
        else:
            raise ValueError(
                "Either neither or both of Xs and Y should be callable, not a mixture"
            )

        dataset_encoded = lambda: itertools.chain.from_iterable(
            map(lambda xy: self.text_to_tokens_mask(*xy), dataset())
        )

        if not callable(Y) and train:
            dataset_encoded_list = list(dataset_encoded())
            class_counts = self._compute_class_counts(dataset_encoded_list)
            self.config.dataset_size = len(dataset_encoded_list)
            if self.config.class_weights is not None:
                self.config.class_weights = compute_class_weights(
                    class_weights=self.config.class_weights, class_counts=class_counts
                )
        shape_def = self.feed_shape_type_def()
        return Dataset.from_generator(
            lambda: self.wrap_tqdm(dataset_encoded(), train), *shape_def
        )

    def _dataset_without_targets(self, Xs, train, context=None):
        if not callable(Xs):
            if context:
                Xs = list(zip(Xs, context))
            Xs_fn = lambda: self.wrap_tqdm(Xs, train)
        else:
            Xs_fn = lambda: self.wrap_tqdm(Xs(), train)

        dataset_encoded = lambda: itertools.chain.from_iterable(
            map(self.text_to_tokens_mask, Xs_fn())
        )
        if not callable(Xs) and self.config.chunk_long_sequences:
            # Adjust dataset size to account for long documents being chunked
            dataset_encoded_list = list(dataset_encoded())
            self.config.dataset_size = len(dataset_encoded_list)
        types, shapes = self.feed_shape_type_def()
        return Dataset.from_generator(
            dataset_encoded, types[0], shapes[0]
        )  # 0s cut out the targets

    def _integer_val_size(self, val_size):
        if isinstance(val_size, float):
            return int(val_size * self.config.dataset_size)
        return val_size

    def validation_settings(self, n_examples, batch_size):
        """
        Auto-select reasonable validation settings
        """
        if self.config.val_size is not None and self.config.val_interval is not None:
            return (
                self._integer_val_size(self.config.val_size),
                self.config.val_interval,
            )

        # Auto-select reasonable validation size
        if self.config.val_size is None:
            if n_examples < 50:
                val_size = 0
            else:
                val_size = max(5, int(0.05 * n_examples))
                val_size = min(100, val_size)
        else:
            val_size = self._integer_val_size(self.config.val_size)

        # Auto-select reasonable validation interval
        if self.config.val_interval is None:
            # sys.maxsize corresponds to never running validation
            # and is used when val_size is set to 0
            val_interval = 4 * int(math.ceil(val_size / batch_size)) or sys.maxsize
        else:
            val_interval = self.config.val_interval

        return int(val_size), int(val_interval)

    def resampling(self, Xs, Y):
        return Xs, Y

    def _make_dataset(self, Xs, Y, context=None, train=False):
        if Y is not None:
            dataset = lambda: self._dataset_with_targets(
                Xs, Y, context=context, train=train
            )
        else:
            dataset = lambda: self._dataset_without_targets(
                Xs, context=context, train=train
            )
        return dataset

    def wrap_tqdm(self, gen, train):
        if self.config.debugging_logs:
            return gen

        if train is None:
            return gen

        try:
            total = len(gen)
        except:
            if train:
                total = self.config.dataset_size
            else:
                total = self.config.val_size

        def internal_gen():
            current_epoch = (self.epoch - 1) % self.config.n_epochs + 1
            it = iter(gen)

            if train:
                if self.config.prefit_init and self.epoch <= self.config.n_epochs:
                    desc = "Initialization Epoch {}/{}".format(
                        current_epoch, self.config.n_epochs
                    )
                else:
                    desc = "Epoch {}/{}".format(current_epoch, self.config.n_epochs)
            else:
                desc = "Validation"
            for _, i in zip(range(self._skip_tqdm), it):
                yield i

            for i in tqdm.tqdm(
                it,
                desc=desc,
                total=total,
                miniters=1,
                leave=current_epoch == self.config.n_epochs and train,
            ):
                yield i

            if train:
                self.epoch += 1

        return internal_gen()

    def get_train_input_fns(
        self, Xs, Y=None, context=None, batch_size=None, val_size=None
    ):
        self.epoch = 1
        batch_size = batch_size or self.config.batch_size

        shuffle_buffer_size = self.config.shuffle_buffer_size
        val_size = val_size or 0
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

        self.config.val_size, self.config.val_interval = self.validation_settings(
            n_examples=len(Xs) if not callable(Xs) else self.config.dataset_size,
            batch_size=batch_size or self.config.batch_size,
        )
        self.config.dataset_size -= val_size

        context_style = None
        if context:
            context_style = context
        if Y is not None:
            self._post_data_initialization(Y=Y, context=context_style)
        else:
            self._post_data_initialization(Y=None, context=context_style)

        if callable(Xs) or Y is None:
            self._skip_tqdm = val_size
            dataset = self._make_dataset(Xs, Y, context, train=True)
            val_dataset_unbatched = (
                lambda: dataset()
                .shuffle(
                    shuffle_buffer_size,
                    seed=self.config.seed,
                    reshuffle_each_iteration=False,
                )
                .take(self.config.val_size)
            )
            train_dataset_unbatched = (
                lambda: dataset()
                .shuffle(
                    shuffle_buffer_size,
                    seed=self.config.seed,
                    reshuffle_each_iteration=False,
                )
                .skip(self.config.val_size)
            )
        else:
            self._skip_tqdm = 0
            if self.config.val_set is None:
                if self.config.val_size == 0:
                    Xs_tr, Xs_va, Y_tr, Y_va, C_tr, C_va = Xs, [], Y, [], context, []
                else:
                    if context:
                        raise FinetuneError(
                            "Validation set with auxiliary info not yet supported."
                        )
                        Xs_tr, Xs_va, Y_tr, Y_va, C_tr, C_va = train_test_split(
                            Xs,
                            Y,
                            context,
                            test_size=self.config.val_size,
                            random_state=self.config.seed,
                        )
                    else:
                        Xs_tr, Xs_va, Y_tr, Y_va = train_test_split(
                            Xs,
                            Y,
                            test_size=self.config.val_size,
                            random_state=self.config.seed,
                        )
                        C_tr, C_va = None, None
            else:
                Xs_tr, Y_tr, C_tr = Xs, Y, context
                if context:
                    Xs_va, Y_va, C_va = self.config.val_set
                else:
                    Xs_va, Y_va, C_va = self.config.val_set

            Xs_tr, Y_tr = self.resampling(Xs_tr, Y_tr)
            self.config.dataset_size = len(Xs_tr)
            val_dataset_unbatched = self._make_dataset(Xs_va, Y_va, C_va, train=False)
            train_dataset_unbatched = self._make_dataset(Xs_tr, Y_tr, C_tr, train=True)

        if self.config.chunk_long_sequences or self.config.class_weights:
            # Certain settings require that the entire dataset be encoded before compiling the graph
            train_dataset_unbatched()

        val_dataset = (
            lambda: val_dataset_unbatched()
            .batch(batch_size, drop_remainder=False)
            .cache()
            .prefetch(prefetch_buffer)
        )
        train_dataset = (
            lambda: train_dataset_unbatched()
            .batch(batch_size, drop_remainder=False)
            .repeat(self.config.n_epochs)
            .prefetch(prefetch_buffer)
        )

        return (
            val_dataset,
            train_dataset,
            self.config.val_size,
            self.config.val_interval,
        )

    def get_predict_input_fn(self, Xs, batch_size=None, context=None):
        batch_size = batch_size or self.config.batch_size
        tf_dataset = lambda: self._dataset_without_targets(
            Xs, train=None, context=context
        ).batch(batch_size)
        return tf_dataset

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
        return [[X]]

    def _format_for_inference(self, X):
        return list(X)

    def _text_to_ids(self, Xs, Y=None, pad_token=None, context=None):
        if context is None and self.config.use_auxiliary_info:
            context = Xs[0]
            Xs = Xs[1]
        Xs = self._format_for_encoding(Xs)
        if self.config.chunk_long_sequences and len(Xs) == 1:
            # can only chunk single sequence inputs
            chunk_size = self.config.max_length - 2
            step_size = chunk_size // 3
            encoded = self.text_encoder.encode_multi_input(
                Xs,
                Y=Y,
                max_length=sys.maxsize,
                pad_token=(pad_token or self.config.pad_token),
                context=context,
            )
            if self.config.use_auxiliary_info:
                processed_context = np.squeeze(
                    self._context_to_vector([encoded.context])
                )

            length = len(encoded.token_ids)
            assert length == len(encoded.token_ids)
            starts = list(range(0, length, step_size))
            for start in starts:
                d = dict()
                end = start + chunk_size
                for field in EncodedOutput._fields:
                    field_value = getattr(encoded, field)
                    if field_value is not None:
                        d[field] = field_value[start:end]
                if self.config.use_auxiliary_info:
                    d["context"] = processed_context[
                        start:end
                    ]  # forced since encoded is immutable'
                else:
                    d['context'] = None

                yield self._array_format(EncodedOutput(**d), pad_token=pad_token)
        else:
            encoder_out = self.text_encoder.encode_multi_input(
                Xs,
                Y=Y,
                max_length=self.config.max_length,
                pad_token=(pad_token or self.config.pad_token),
                context=context,
            )

            d = dict()
            for field in EncodedOutput._fields:
                field_value = getattr(encoder_out, field)
                if field_value is not None:
                    d[field] = field_value
            if self.config.use_auxiliary_info:
                d["context"] = np.squeeze(
                    self._context_to_vector([encoder_out.context])
                )  # forced since encoded is immutable
            else:
                d["context"] = None

            yield self._array_format(
                EncodedOutput(**d), pad_token=(pad_token or self.config.pad_token)
            )


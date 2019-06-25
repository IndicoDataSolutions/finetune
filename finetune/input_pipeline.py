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
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import finetune
from finetune.errors import FinetuneError
from finetune.encoding.input_encoder import ArrayEncodedOutput, EncodedOutput
from finetune.util.imbalance import compute_class_weights

LOGGER = logging.getLogger('finetune')


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        self.text_encoder = self.config.base_model.get_encoder()
        self.label_encoder = None
        self.context_encoder = MultiLabelBinarizer()
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
        return (
            (
                {
                    "tokens": tf.int32,
                    "mask": tf.float32
                },
                tf.float32
            ),
            (
                {
                    "tokens": TS([self.config.max_length, 2]),
                    "mask": TS([self.config.max_length])
                },
                TS([self.target_dim])
            )
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
        mask = np.zeros((self.config.max_length), dtype=np.float32)

        if encoded_output.labels is not None:
            print(encoded_output.labels)
            print(len(encoded_output.labels))
            labels_arr = np.empty((self.config.max_length), dtype='object')
            labels_arr.fill((pad_token or self.config.pad_token))
        else:
            labels_arr = None

        if encoded_output.context is not None:
            print(encoded_output.context)
            print(len(encoded_output.context))
            context_arr = np.empty((self.config.max_length), dtype='object')
            context_arr.fill((pad_token or self.config.pad_token))

        # BPE embedding
        x[:seq_length, 0] = encoded_output.token_ids
        # masking: value of 1 means "consider this in cross-entropy LM loss"
        mask[1:seq_length] = 1
        if encoded_output.labels:
            labels_arr[:seq_length] = encoded_output.labels
        if encoded_output.context:
            context_arr[:seq_length] = encoded_output.context
        # positional_embeddings
        x[:, 1] = np.arange(
            self.text_encoder.vocab_size, self.text_encoder.vocab_size + self.config.max_length
        )

        output = ArrayEncodedOutput(
            token_ids=x,
            tokens=encoded_output.tokens,
            labels=labels_arr,
            char_locs=encoded_output.char_locs,
            mask=mask,
            context=context_arr
        )
        return output

    def text_to_tokens_mask(self, X, Y=None, context=None):
        out_gen = self._text_to_ids(X, pad_token=self.config.pad_token)
        for out in out_gen:
            feats = {"tokens": out.token_ids, "mask": out.mask, "context": out.context}
            output = [feats]
            if context is not None:
                output.append(context)
            if Y is not None:
                output.append(self.label_encoder.transform([Y])[0])
            yield output

    def _post_data_initialization(self, Y=None, context=None):
        if Y:
            self.label_encoder = self._target_encoder()
            if not callable(Y):
                Y_fit = Y
                self.label_encoder.fit(Y)
            else:
                Y_fit = list(itertools.islice(Y(), 10000))
                self.label_encoder.fit(Y_fit)
            self.config.pad_idx = self.pad_idx

            target_dim = self.label_encoder.target_dim
            self.lm_loss_coef = self.config.lm_loss_coef if target_dim is not None else 1.0
            self.target_dim = target_dim

        if context:
            if not callable(Y):
                characteristics = context
            else:
                characteristics = itertools.islice(context(), 10000)
            num_samples = len(characteristics)
            _ = self._context_to_vector(characteristics) # to fit the auxiliary information label encoders

    def _context_to_vector(self, context):
            num_samples = len(context)
            characteristics = itertools.chain.from_iterable(context)
            characteristics = pd.DataFrame(characteristics).to_dict('list')

            if not hasattr(self, 'label_encoders'): # we will fit necessary label encoders here to know dimensionality of input vector before entering featurizer
                self.label_encoders = {k:LabelBinarizer() for k in characteristics.keys() if 
                all(not type(data) == int and not type(data) == float for data in characteristics[k])} # Excludes features that are continuous, like position and font size

                for label, encoder in self.label_encoders.items():
                    encoder.fit(characteristics[label])

                self.context_labels = sorted(characteristics.keys()) # sort for consistent ordering between runs
                continuous_labels = [label for label in self.context_labels if label not in self.label_encoders.keys()]
                self.context_dim = len(continuous_labels) + np.sum([len(encoder.classes_) for encoder in self.label_encoders.values()]) #since categorical labels use LabelBinarizer, they take num_possible_labels dimensions each for one-hot-encoding, while numerical features only use 1 dimension
                self.config.context_dim = self.context_dim

            # make sure all labels have a value for every token, and calculate total num tokens
            num_tokens = None
            for label in self.context_labels:
                new_length = len(characteristics[label]) if type(characteristics[label] == list) else 1
                if num_tokens is not None and num_tokens != new_length:
                    raise FinetuneError('Incorrect label shapes.')
                num_tokens = new_length

            vector = np.zeros((num_tokens, self.context_dim)) # final shape: (num_tokens, self.context_dim)
            #print(np.shape(vector))
            current_index = 0
            for label in self.context_labels:
                #print(label)
                data = characteristics[label]
                if label in self.label_encoders.keys():
                    data = self.label_encoders[label].transform(data)
                    data_dim = len(self.label_encoders[label].classes_)
                    if data_dim == 2: # since binary classes default to column vector
                        data_dim=1
                    #print(self.label_encoders[label].classes_)
                else:
                    #data = [data]
                    data_dim = 1
                #print(data)
                #print("DATA DIM:"+str(data_dim))
                for sample_idx in range(num_tokens):
                    for label_dimension in range(data_dim):
                        #print("Label dim:" + str(label_dimension) + "Sample:" + str(sample_idx))
                        #print(data[sample_idx])
                        vector[sample_idx][current_index + label_dimension] = data[sample_idx] if data_dim  == 1 else data[sample_idx][label_dimension]
                current_index += 1

            return vector

    def _compute_class_counts(self, encoded_dataset):
        target_arrs = np.asarray([target_arr for doc, target_arr in encoded_dataset])
        return Counter(self.label_encoder.inverse_transform(target_arrs))
        
    def _dataset_with_targets(self, Xs, Y, train, context=None):
        print(Y)
        if not callable(Xs) and not callable(Y):
            dataset = lambda: zip(Xs, Y, context) # Do not need to check if context is callable - it is turned in along with Xs, and thus must have the same form
        elif callable(Xs) and callable(Y):
            dataset = lambda: zip(Xs(), Y(), context())  # encode one sample at a time.
        else:
            raise ValueError("Either neither or both of Xs and Y should be callable, not a mixture")

        dataset_encoded = lambda: itertools.chain.from_iterable(
            map(lambda xy: self.text_to_tokens_mask(*xy), dataset()))
        shape_def = self.feed_shape_type_def()

        if not callable(Y) and train:
            dataset_encoded_list = list(dataset_encoded())
            class_counts = self._compute_class_counts(dataset_encoded_list)
            self.config.dataset_size = len(dataset_encoded_list)
            if self.config.class_weights is not None:
                self.config.class_weights = compute_class_weights(
                    class_weights=self.config.class_weights, 
                    class_counts=class_counts
                )
           
        return Dataset.from_generator(lambda: self.wrap_tqdm(dataset_encoded(), train), *shape_def)

    def _dataset_without_targets(self, Xs, train):
        if not callable(Xs):
            Xs_fn = lambda: self.wrap_tqdm(Xs, train)
        else:
            Xs_fn = lambda: self.wrap_tqdm(Xs(), train)

        dataset_encoded = lambda: itertools.chain.from_iterable(map(self.text_to_tokens_mask, Xs_fn()))
        if not callable(Xs) and self.config.chunk_long_sequences:
            # Adjust dataset size to account for long documents being chunked
            dataset_encoded_list = list(dataset_encoded())
            self.config.dataset_size = len(dataset_encoded_list)
        types, shapes = self.feed_shape_type_def()
        return Dataset.from_generator(dataset_encoded, types[0], shapes[0])  # 0s cut out the targets

    def _integer_val_size(self, val_size):
        if isinstance(val_size, float):
            return int(val_size * self.config.dataset_size)
        return val_size

    def validation_settings(self, n_examples, batch_size):
        """
        Auto-select reasonable validation settings
        """
        if self.config.val_size is not None and self.config.val_interval is not None:
            return self._integer_val_size(self.config.val_size), self.config.val_interval

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
            dataset = lambda: self._dataset_with_targets(Xs, Y, context=context, train=train)
        else:
            dataset = lambda: self._dataset_without_targets(Xs, context=context, train=train)
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
                    desc = "Initialization Epoch {}/{}".format(current_epoch, self.config.n_epochs)
                else:
                    desc = "Epoch {}/{}".format(current_epoch, self.config.n_epochs)
            else:
                desc = "Validation"
            for _, i in zip(range(self._skip_tqdm), it):
                yield i

            for i in tqdm.tqdm(it, desc=desc, total=total, miniters=1, leave=current_epoch == self.config.n_epochs and train):
                yield i

            if train:
                self.epoch += 1

        return internal_gen()

    def get_train_input_fns(self, Xs, Y=None, context=None, batch_size=None, val_size=None):
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
            batch_size=batch_size or self.config.batch_size
        )
        self.config.dataset_size -= val_size

        if Y is not None:
            self._post_data_initialization(Y=Y, context=context)
        else:
            self._post_data_initialization(context=context)

        if callable(Xs) or Y is None:
            self._skip_tqdm = val_size
            dataset = self._make_dataset(Xs, Y, context, train=True)
            val_dataset_unbatched = lambda: dataset().shuffle(
                shuffle_buffer_size, seed=self.config.seed, reshuffle_each_iteration=False
            ).take(self.config.val_size)
            train_dataset_unbatched = lambda: dataset().shuffle(
                shuffle_buffer_size, seed=self.config.seed, reshuffle_each_iteration=False
            ).skip(self.config.val_size)
        else:
            self._skip_tqdm = 0
            if self.config.val_set is None:
                if self.config.val_size == 0:
                    Xs_tr, Xs_va, Y_tr, Y_va = Xs, [], Y, []
                else:
                    Xs_tr, Xs_va, Y_tr, Y_va = train_test_split(Xs, Y, context, test_size=self.config.val_size, random_state=self.config.seed)
            else:
                Xs_tr, Y_tr = Xs, Y
                Xs_va, Y_va = self.config.val_set

            Xs_tr, Y_tr = self.resampling(Xs_tr, Y_tr)
            self.config.dataset_size = len(Xs_tr)
            val_dataset_unbatched = self._make_dataset(Xs_va, Y_va, context, train=False)
            train_dataset_unbatched = self._make_dataset(Xs_tr, Y_tr, context, train=True)

        if self.config.chunk_long_sequences or self.config.class_weights:
            # Certain settings require that the entire dataset be encoded before compiling the graph
            train_dataset_unbatched()

        val_dataset = lambda: val_dataset_unbatched().batch(batch_size, drop_remainder=False).cache().prefetch(prefetch_buffer)
        train_dataset = lambda: train_dataset_unbatched().batch(batch_size, drop_remainder=False).repeat(
            self.config.n_epochs).prefetch(prefetch_buffer)

        return val_dataset, train_dataset, self.config.val_size, self.config.val_interval

    def get_predict_input_fn(self, Xs, batch_size=None):
        batch_size = batch_size or self.config.batch_size
        tf_dataset = lambda: self._dataset_without_targets(Xs, train=None).batch(batch_size)
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
        Xs = self._format_for_encoding(Xs)
        if self.config.chunk_long_sequences and len(Xs) == 1:

            print(Y)
            # can only chunk single sequence inputs
            chunk_size = self.config.max_length - 2
            step_size = chunk_size // 3
            encoded = self.text_encoder.encode_multi_input(
                Xs,
                Y=Y,
                max_length=sys.maxsize,
                pad_token=(pad_token or self.config.pad_token),
                context=context
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
            encoder_out = self.text_encoder.encode_multi_input(
                Xs,
                Y=Y,
                max_length=self.config.max_length,
                pad_token=(pad_token or self.config.pad_token),
                context=context
            )

            yield self._array_format(encoder_out, pad_token=(pad_token or self.config.pad_token))
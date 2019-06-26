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

        print('NUM TOKENS')
        print(seq_length)
        print('NUM CONTEXT')
        print(len(np.squeeze(encoded_output.context)))
        print('NUM LABEL')
        print(len(encoded_output.labels))
        

        if encoded_output.labels is not None:
            labels_arr = np.empty((self.config.max_length), dtype='object')
            labels_arr.fill((pad_token or self.config.pad_token))
        else:
            labels_arr = None

        if encoded_output.context is not None:
            context_arr = np.empty((self.config.max_length, self.config.context_dim), dtype='object')
            context_arr.fill((pad_token or self.config.pad_token))

        # BPE embedding
        x[:seq_length, 0] = encoded_output.token_ids
        # masking: value of 1 means "consider this in cross-entropy LM loss"
        mask[1:seq_length] = 1
        if encoded_output.labels:
            labels_arr[:seq_length] = encoded_output.labels
        if encoded_output.context:
            context_arr[:seq_length][:] = np.squeeze(encoded_output.context)

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
            """
            Takes list of context dictionaries and turns them into lists (per sample) of lists (per token) of context vectors
            """
            num_samples = len(context)
            vector_list = []

            if not hasattr(self, 'label_encoders'): # we will fit necessary label encoders here to know dimensionality of input vector before entering featurizer
                characteristics = itertools.chain.from_iterable(context)
                characteristics = pd.DataFrame(characteristics).to_dict('list')

                self.label_encoders = {k:LabelBinarizer() for k in characteristics.keys() if 
                all(k != self.config.pad_token and not (type(data) == int or (type(data) == float and not pd.isna(data))) for data in characteristics[k])} # Excludes features that are continuous, like position and font size, since they do not have categorical binary encodings
                for label, encoder in self.label_encoders.items():
                    without_nans = [x for x in characteristics[label] if not pd.isna(x)]
                    encoder.fit(without_nans)

                self.context_labels = sorted(characteristics.keys()) # sort for consistent ordering between runs
                continuous_labels = [label for label in self.context_labels if label not in self.label_encoders.keys()]

                self.context_dim = len(continuous_labels) # Sum over features to determine dimensionality of each feature vector.
                for encoder in self.label_encoders.values(): # since categorical labels use LabelBinarizer, they use varying dimensions each for each one-hot-encoding, while numerical features only use 1 dimension, so we sum over all for the total dimension.
                    self.context_dim += 1 if len(encoder.classes_) == 1 else len(encoder.classes_) # Binary encodings with only two classes are given with one bit. Encodings with n > 1 classes are given with n bits. (Thanks sklearn)
                self.config.context_dim = self.context_dim
                return True

            for sample in context:
                #sample = sample[1:len(context)-2] # remove special tokens
                print(sample)
                # See which tokens have padded labels/context vectors, and thus have a zero vector for features
                padded_indices = []
                tokens_with_context=[]
                for idx in range(len(sample)):
                    if type(sample[idx]) == str and sample[idx] == self.config.pad_token:
                        padded_indices.append(idx)
                    else:
                        tokens_with_context.append(sample[idx])

                #if len(sample) > 1:
                characteristics = pd.DataFrame(tokens_with_context).to_dict('list')
                #else:
                #    characteristics = {k:[v] for k,v in sample.pop().items()}
                print(characteristics)
                print("padded indices:")
                print(padded_indices)

                # make sure all features cover the same number of tokens, and calculate total num tokens
                num_tokens = None
                for label in self.context_labels:
                    new_length = len(characteristics[label]) if type(characteristics[label] == list) else 1
                    if num_tokens is not None and num_tokens != new_length:
                        raise FinetuneError('Incorrect label shapes.')
                    num_tokens = new_length
                
                print("INPUT PIPELINE NUM TOKENS")
                print(num_tokens)

                vector = np.zeros((num_tokens + len(padded_indices), self.config.context_dim)) # Feature vector for one document. Add 2 for the special tokens at beginning/end
                current_index = 0

                # Loop through each feature and add each to new index of the feature vector
                for label in self.context_labels:
                    if label == self.config.pad_token:
                        continue
                    data = characteristics[label]

                    # Binary encoded features have different dimensionality as simple floats/ints, so must be handled differently.
                    if label in self.label_encoders.keys():
                        without_nans = [x for x in data if not pd.isna(x)]
                        data = self.label_encoders[label].transform(without_nans) #this removes nans from padded tokens, but now the list is too short. The 'num_backward' variable will track this offset to ensure indices are correctly tracked.
                        data_dim = len(self.label_encoders[label].classes_)
                        if data_dim == 2: # since binary classes default to column vector
                            data_dim=1
                    else:
                        data = [x for x in data if not pd.isna(x)]
                        data_dim = 1

                    #loop through indices and fill with correct data
                    num_backward = 0
                    for sample_idx in range(num_tokens):
                        if sample_idx in padded_indices: # there is no data, simply a pad from the encoder, so fill out with zero vector
                            vector[sample_idx][:] = 0
                            num_backward += 1 #since we're skipping this sample, the next sample needs to be filled from this sample's index in data. This variable tracks how far back we need to go for following indices
                            continue
                        for label_dimension in range(data_dim):
                            #print("Label dim:" + str(label_dimension + current_index) + "Sample Index:" + str(sample_idx) + "Current label" + label)
                            #if self.config.pad_token in sample[sample_idx]: 
                            #    vector[sample_idx][current_index + label_dimension] = 0
                            #else:
                            vector[sample_idx][current_index + label_dimension] = data[sample_idx - num_backward] if data_dim  == 1 else data[sample_idx - num_backward][label_dimension]

                    current_index += 1
                vector_list.append(vector)
            print('Vector shape')
            print(np.shape(vector_list))
            return vector_list



    def _compute_class_counts(self, encoded_dataset):
        target_arrs = np.asarray([target_arr for doc, target_arr in encoded_dataset])
        return Counter(self.label_encoder.inverse_transform(target_arrs))
        
    def _dataset_with_targets(self, Xs, Y, train, context=None):
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
                    Xs_tr, Xs_va, Y_tr, Y_va, C_tr, C_va = Xs, [], Y, [], context, []
                else:
                    if context:
                        Xs_tr, Xs_va, Y_tr, Y_va, C_tr, C_va = train_test_split(Xs, Y, context, test_size=self.config.val_size, random_state=self.config.seed)
                    else:
                        Xs_tr, Xs_va, Y_tr, Y_va = train_test_split(Xs, Y, test_size=self.config.val_size, random_state=self.config.seed)
                        C_tr, C_va = None
            else:
                Xs_tr, Y_tr = Xs, Y
                Xs_va, Y_va = self.config.val_set

            Xs_tr, Y_tr = self.resampling(Xs_tr, Y_tr)
            self.config.dataset_size = len(Xs_tr)

            print(self.config.dataset_size)
            val_dataset_unbatched = self._make_dataset(Xs_va, Y_va, C_va, train=False)
            train_dataset_unbatched = self._make_dataset(Xs_tr, Y_tr, C_tr, train=True)

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
            processed_context = self._context_to_vector([encoded.context])
            print(np.shape(processed_context))
            length = len(encoded.token_ids)
            starts = list(range(0, length, step_size))
            for start in starts:
                d = dict()
                end = start + chunk_size
                for field in EncodedOutput._fields:
                    field_value = getattr(encoded, field)
                    if field_value is not None:
                        d[field] = field_value[start:end]
                d['context'] = processed_context[start:end] # forced since encoded is immutable
                yield self._array_format(EncodedOutput(**d), pad_token=pad_token)
        else:
            encoder_out = self.text_encoder.encode_multi_input(
                Xs,
                Y=Y,
                max_length=self.config.max_length,
                pad_token=(pad_token or self.config.pad_token)
            )

            yield self._array_format(encoder_out, pad_token=(pad_token or self.config.pad_token))
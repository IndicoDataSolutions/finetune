import numpy as np

from finetune.base import BaseModel, CLASSIFICATION, REGRESSION, SEQUENCE_LABELING
from finetune.errors import InvalidTargetType
import tensorflow as tf


class Comparison(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _text_to_ids(self, X1, X2, max_length=None):
        max_length = max_length or self.config.max_length
        forward_pairs = self.encoder.encode_multi_input(X1, X2, max_length=max_length, verbose=self.config.verbose)
        backward_pairs = self.encoder.encode_multi_input(X2, X1, max_length=max_length, verbose=self.config.verbose)
        seq_array_fw = self._array_format(forward_pairs)
        seq_array_bw = self._array_format(backward_pairs)
        token_ids = np.stack([seq_array_fw.token_ids, seq_array_bw.token_ids], 1)
        mask = np.stack([seq_array_fw.mask, seq_array_bw.mask], 1)

        return token_ids, mask

    def finetune(self, X1, X2, Y, batch_size=None):
        """
        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        if self.target_type is None:
            if np.array(Y).dtype == 'float':
                self.target_type = REGRESSION
            elif len(Y.shape) == 1:  # [batch]
                self.target_type = CLASSIFICATION
            else:
                raise InvalidTargetType(
                    "targets must either be a 1-d array of classification targets or a "
                    "2-d array of sequence labels."
                )

        return self._finetune(X1, X2, Y=Y, batch_size=batch_size)

    def _define_placeholders(self):
        super()._define_placeholders()
        self.X = tf.placeholder(tf.int32, [None, 2, self.config.max_length, 2])
        self.M = tf.placeholder(tf.float32, [None, 2, self.config.max_length])  # sequence mask

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        featurizer_state["sequence_features"] = tf.reduce_sum(featurizer_state["sequence_features"], 1)
        featurizer_state["features"] = tf.reduce_sum(featurizer_state["features"], 1)
        return super()._target_model(featurizer_state, targets, n_outputs, train, reuse, **kwargs)

    def predict(self, X1, X2, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return self._predict(X1, X2, max_length=max_length)

    def predict_proba(self, X1, X2, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return self._predict_proba(X1, X2, max_length=max_length)

    def featurize(self, X1, X2, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X1, X2, max_length=max_length)

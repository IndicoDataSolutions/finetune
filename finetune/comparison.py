import numpy as np

from finetune.base import BaseModel
from finetune.classifier import Classifier
from finetune.encoding import ArrayEncodedOutput
import tensorflow as tf


class Comparison(Classifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _text_to_ids(self, X1, X2, Y=None, max_length=None):
        """
        Format comparison examples as a list of IDs
        """
        arr_forward = super()._text_to_ids(X1, X2, Y=Y, max_length=max_length)
        arr_backward = super()._text_to_ids(X2, X1, Y=Y, max_length=max_length)
        kwargs = arr_forward._asdict()
        kwargs['tokens'] = [arr_forward.tokens, arr_backward.tokens]
        kwargs['token_ids'] = np.stack([arr_forward.token_ids, arr_backward.token_ids], 1)
        kwargs['mask'] = np.stack([arr_forward.mask, arr_backward.mask], 1)
        return ArrayEncodedOutput(**kwargs)

    def finetune(self, X1, X2, Y, batch_size=None):
        """
        :param X1: List or array of text, shape [batch]
        :param X2: List or array of text, shape [batch]
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        fit_language_model_only = (Y is None)
        arr_encoded = self._text_to_ids(X1, X2)
        labels = None if fit_language_model_only else Y
        return self._training_loop(arr_encoded, Y=labels, batch_size=batch_size)

    def _define_placeholders(self):
        super()._define_placeholders()
        self.X = tf.placeholder(tf.int32, [None, 2, self.config.max_length, 2])
        self.M = tf.placeholder(tf.float32, [None, 2, self.config.max_length])  # sequence mask

    def _target_model(self, *, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        featurizer_state["sequence_features"] = tf.reduce_sum(featurizer_state["sequence_features"], 1)
        featurizer_state["features"] = tf.reduce_sum(featurizer_state["features"], 1)
        return super()._target_model(featurizer_state=featurizer_state, targets=targets, n_outputs=n_outputs, train=train, reuse=reuse, **kwargs)

    def predict(self, X1, X2, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param X1: List or array of text, shape [batch]
        :param X2: List or array of text, shape [batch]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return BaseModel.predict(self, X1, X2, max_length=max_length)

    def predict_proba(self, X1, X2, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.


        :param X1: List or array of text, shape [batch]
        :param X2: List or array of text, shape [batch]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return BaseModel.predict(self, X1, X2, max_length=max_length)

    def featurize(self, X1, X2, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X1: List or array of text, shape [batch]
        :param X2: List or array of text, shape [batch]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, X1, X2, max_length=max_length)

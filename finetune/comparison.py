import numpy as np

from finetune.base import BaseModel
from finetune.classifier import Classifier
from finetune.encoding import ArrayEncodedOutput
import tensorflow as tf


class Comparison(Classifier):
    """ 
    Compares two documents to solve a classification task.  
    
    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _text_to_ids(self, pairs, Y=None, max_length=None):
        """
        Format comparison examples as a list of IDs
        
        pairs: Array of text, shape [batch, 2]
        """
        arr_forward = super()._text_to_ids(pairs, Y=Y, max_length=max_length)
        reversed_pairs = [pair[::-1] for pair in pairs]
        arr_backward = super()._text_to_ids(reversed_pairs, Y=Y, max_length=max_length)
        kwargs = arr_forward._asdict()
        kwargs['tokens'] = [arr_forward.tokens, arr_backward.tokens]
        kwargs['token_ids'] = np.stack([arr_forward.token_ids, arr_backward.token_ids], 1)
        kwargs['mask'] = np.stack([arr_forward.mask, arr_backward.mask], 1)
        return ArrayEncodedOutput(**kwargs)

    def finetune(self, pairs, Y=None, batch_size=None):
        """
        :param pairs: Array of text, shape [batch, 2]
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        arr_encoded = self._text_to_ids(pairs, Y=Y)
        return self._training_loop(arr_encoded, Y=Y, batch_size=batch_size)

    def _define_placeholders(self, target_dim=None):
        super()._define_placeholders(target_dim=target_dim)
        self.X = tf.placeholder(tf.int32, [None, 2, self.config.max_length, 2])
        self.M = tf.placeholder(tf.float32, [None, 2, self.config.max_length])  # sequence mask

    def _target_model(self, *, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        featurizer_state["sequence_features"] = tf.abs(tf.reduce_sum(featurizer_state["sequence_features"], 1))
        featurizer_state["features"] = tf.abs(tf.reduce_sum(featurizer_state["features"], 1))
        return super()._target_model(featurizer_state=featurizer_state, targets=targets, n_outputs=n_outputs, train=train, reuse=reuse, **kwargs)

    def predict(self, pairs, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param pairs: Array of text, shape [batch, 2]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return BaseModel.predict(self, pairs, max_length=max_length)

    def predict_proba(self, pairs, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.


        :param pairs: Array of text, shape [batch, 2]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return BaseModel.predict_proba(self, pairs, max_length=max_length)

    def featurize(self, pairs, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param pairs: Array of text, shape [batch, 2]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, pairs, max_length=max_length)

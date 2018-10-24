import numpy as np
import tensorflow as tf

from finetune.base import BaseModel
from finetune.classifier import Classifier, ClassificationPipeline
from finetune.encoding import ArrayEncodedOutput


class ComparisonPipeline(ClassificationPipeline):

    def _format_for_encoding(self, X):
        return [X]


    def _text_to_ids(self, pair, Y=None, pad_token=None):
        """
        Format comparison examples as a list of IDs

        pairs: Array of text, shape [batch, 2]
        """
        assert self.config.chunk_long_sequences is False, "Chunk Long Sequences is not compatible with comparison"
        arr_forward = next(super()._text_to_ids(pair, Y=None))
        reversed_pair = pair[::-1]
        arr_backward = next(super()._text_to_ids(reversed_pair, Y=None))
        kwargs = arr_forward._asdict()
        kwargs['tokens'] = [arr_forward.tokens, arr_backward.tokens]
        kwargs['token_ids'] = np.stack([arr_forward.token_ids, arr_backward.token_ids], 0)
        kwargs['mask'] = np.stack([arr_forward.mask, arr_backward.mask], 0)
        yield ArrayEncodedOutput(**kwargs)

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        return ({"tokens": tf.int32, "mask": tf.int32}, tf.int32), (
            {"tokens": TS([2, self.config.max_length, 2]), "mask": TS([2, self.config.max_length])},
            TS([self.target_dim]))



class Comparison(Classifier):
    """ 
    Compares two documents to solve a classification task.  
    
    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_input_pipeline(self):
        return ComparisonPipeline(self.config)

    def _target_model(self, *, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        featurizer_state["sequence_features"] = tf.abs(tf.reduce_sum(featurizer_state["sequence_features"], 1))
        featurizer_state["features"] = tf.abs(tf.reduce_sum(featurizer_state["features"], 1))
        return super()._target_model(featurizer_state=featurizer_state, targets=targets, n_outputs=n_outputs, train=train, reuse=reuse, **kwargs)

    def predict(self, pairs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param pairs: Array of text, shape [batch, 2]
        :returns: list of class labels.
        """
        return BaseModel.predict(self, pairs)

    def predict_proba(self, pairs):
        """
        Produces a probability distribution over classes for each example in X.


        :param pairs: Array of text, shape [batch, 2]
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return BaseModel.predict_proba(self, pairs)

    def featurize(self, pairs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param pairs: Array of text, shape [batch, 2]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, pairs)

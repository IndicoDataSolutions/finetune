import warnings

import tensorflow as tf

from finetune.base import BaseModel
from finetune.target_encoders import MultilabelClassificationEncoder
from finetune.network_modules import multi_classifier

from finetune.input_pipeline import BasePipeline


class MultilabelClassificationPipeline(BasePipeline):
    def _target_encoder(self):
        return MultilabelClassificationEncoder()


class MultiLabelClassifier(BaseModel):
    """ 
    Classifies a single document into upto N of N categories.
    
    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_placeholder = None

    def _get_input_pipeline(self):
        return MultilabelClassificationPipeline(self.config)

    def featurize(self, X):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X)

    def predict(self, X, threshold=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        self.config._threshold = threshold or self.config.multi_label_threshold
        return self._predict(X)

    def predict_proba(self, X):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return super().predict_proba(X)

    def finetune(self, X, Y=None, batch_size=None):
        """
        :param X: list or array of text.
        :param Y: A list of lists containing labels for the corresponding X
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=Y, batch_size=batch_size)

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return multi_classifier(
            hidden=featurizer_state['features'],
            targets=targets,
            n_targets=n_outputs,
            config=self.config,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        threshold = kwargs.get("threshold", self.config.multi_label_threshold)
        return tf.cast(tf.nn.sigmoid(logits) > threshold, tf.int32)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.sigmoid(logits)
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import RandomOverSampler

from finetune.base import BaseModel
from finetune.target_encoders import OneHotLabelEncoder
from finetune.network_modules import classifier
from finetune.input_pipeline import BasePipeline


class ClassificationPipeline(BasePipeline):

    def resampling(self, Xs, Y):
        if self.config.oversample:
            idxs, Ys = RandomOverSampler().fit_sample([[i] for i in range(len(Xs))], Y)
            return [Xs[i[0]] for i in idxs], [Ys[i[0]] for i in idxs]
        return Xs, Y

    def _target_encoder(self):
        return OneHotLabelEncoder()


class Classifier(BaseModel):
    """ 
    Classifies a single document into 1 of N categories.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def _get_input_pipeline(self):
        return ClassificationPipeline(self.config)

    def featurize(self, X):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X)

    def predict(self, X):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        return super().predict(X)

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
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=Y, batch_size=batch_size)

    def get_eval_fn(cls):
        return lambda labels, targets: np.mean(np.asarray(labels) == np.asarray(targets))

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return classifier(
            hidden=featurizer_state['features'], 
            targets=targets, 
            n_targets=n_outputs, 
            config=self.config,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        return tf.argmax(logits, -1)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

import tensorflow as tf
import numpy as np

from finetune.encoding.target_encoders import RegressionEncoder
from finetune.encoding.input_encoder import ArrayEncodedOutput
from finetune.target_models.comparison import ComparisonPipeline
from finetune.nn.target_blocks import regressor
from finetune.base import BaseModel
from finetune.nn.auxiliary import add_context_embed


class ComparisonRegressionPipeline(ComparisonPipeline):

    def _target_encoder(self):
        return RegressionEncoder()

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32, "mask": tf.int32}
        shapes = {"tokens": TS([2, None, 2]), "mask": TS([2, None])}
        if self.config.use_auxiliary_info:
            TS = tf.TensorShape
            types["context"] = tf.float32
            shapes["context"] = TS([2, None, self.config.context_dim])
        return (
            (types, tf.float32,),
            (shapes, TS([self.target_dim]),),
        )


class ComparisonRegressor(BaseModel):
    """ 
    Compares two documents to solve a regression task.  
    
    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """
    defaults = {"chunk_long_sequences": False}
    
    def _get_input_pipeline(self):
        return ComparisonRegressionPipeline(self.config)

    
    def _pre_target_model_hook(self, featurizer_state):
        add_context_embed(featurizer_state)
        featurizer_state["sequence_features"] = tf.abs(tf.reduce_sum(featurizer_state["sequence_features"], 1))
        featurizer_state["features"] = tf.abs(tf.reduce_sum(featurizer_state["features"], 1))

    def _target_model(self, *, config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return regressor(
            hidden=featurizer_state['features'],
            targets=targets, 
            n_targets=n_outputs,
            config=config,
            train=train, 
            reuse=reuse, 
            **kwargs
        )

    def predict(self, pairs, context=None, **kwargs):
        """
        Produces a floating point prediction determined by the fine-tuned model.


        :param pairs: Array of text, shape [batch, 2]
        :returns: list of floats, shape [batch]
        """
        return super().predict(pairs, context=context, **kwargs)

    def predict_proba(self, pairs, context=None):
        """
        Not implemented in regression task.
        """
        raise AttributeError("`ComparisonRegressor` model does not support `predict_proba`.")

    def featurize(self, pairs, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param pairs: Array of text, shape [batch, 2]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(pairs, **kwargs)
    
    
    def finetune(self, pairs, Y=None, batch_size=None, context=None, **kwargs):
        """
        :param pairs: Array of text, shape [batch_size, 2]
        :param Y: floating point targets
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(pairs, Y=Y, batch_size=batch_size, context=context, **kwargs)
    
    def _predict_op(self, logits, **kwargs):
        return logits

    def _predict_proba_op(self, logits, **kwargs):
        return logits

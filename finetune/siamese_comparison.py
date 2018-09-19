import numpy as np

from finetune.comparison import Comparison
from finetune.encoding import ArrayEncodedOutput
from finetune.network_modules import cosine_similarity
from finetune.base import BaseModel
import tensorflow as tf


class SiameseComparison(Comparison):
    """
    Compares two documents to via a siamese network and produces a similarity score (between 0-1).

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """


    def _text_to_ids(self, pairs, Y=None):
        """
        Format comparison examples as a list of IDs

        pairs: Array of text, shape [batch, 2]
        """
        arr_0 = BaseModel._text_to_ids(self, [pair[0] for pair in pairs], Y=Y)
        arr_1 = BaseModel._text_to_ids(self, [pair[1] for pair in pairs], Y=Y)
        kwargs = arr_0._asdict()
        kwargs['tokens'] = [arr_0.tokens, arr_1.tokens]
        kwargs['token_ids'] = np.stack([arr_0.token_ids, arr_1.token_ids], 1)
        kwargs['mask'] = np.stack([arr_0.mask, arr_1.mask], 1)
        return ArrayEncodedOutput(**kwargs)

    def _define_placeholders(self, target_dim=None):
        super()._define_placeholders(target_dim=target_dim)
        self.X = tf.placeholder(tf.int32, [None, 2, self.config.max_length, 2])
        self.M = tf.placeholder(tf.float32, [None, 2, self.config.max_length])  # sequence mask

    def _target_model(self, *, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        hidden_0, hidden_1 = tf.split(featurizer_state["features"], num_or_size_splits=2, axis=1)
        hidden_0 = tf.squeeze(hidden_0, axis=[1])
        hidden_1 = tf.squeeze(hidden_1, axis=[1])
        return cosine_similarity(
            hidden_0=hidden_0, 
            hidden_1=hidden_1, 
            targets=targets, 
            n_targets=n_outputs,
            train=train,
            reuse=reuse,
            dropout_placeholder=self.do_dropout,
            config=self.config,
            **kwargs
        )
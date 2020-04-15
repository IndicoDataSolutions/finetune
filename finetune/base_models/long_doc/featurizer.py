import tensorflow as tf

from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.gpt.featurizer import dropout, embed
from finetune.base_models.tcn.featurizer import TemporalBlock


def long_doc_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    **kwargs
):
    """
    The featurizer element of the finetuning model. Maps from tokens ids to a dense embedding of the sequence.

    :param X: A tensor of token indexes with shape [batch_size, sequence_length, token_idx]
    :param encoder: A TextEncoder object.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :return: A dict containing;
        embed_weights: the word embedding matrix.
        features: The output of the featurizer_final state.
        sequence_features: The output of the featurizer at each timestep.
    """
    initial_shape = tf.shape(X)
    print(X.get_shape().as_list())
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-2:]), 0))
    print(X.get_shape().as_list())
    sequence_length = tf.shape(X)[1]
    with tf.variable_scope("model/featurizer", reuse=reuse):
        h = X # X is already embedded

        clf_token = encoder.start_token

        with tf.variable_scope("tcn_stack"):
            representation = h
            for layer_num in range(config.n_layer):
                representation = TemporalBlock(
                    n_filters=config.n_filter,
                    kernel_size=config.kernel_size,
                    rate=config.resid_p_drop if train else 0,
                    dilation_rate=2 ** layer_num,
                    scope="Temporal{}".format(layer_num),
                )(representation)

        seq_feats = tf.reshape(representation, shape=[-1, sequence_length, config.n_filter])
        print(seq_feats.get_shape().as_list())

        # mask out the values past the classify token before performing pooling
        pool_idx = tf.cast(
            tf.argmax(tf.cast(tf.equal(X, clf_token), tf.float32), 1),
            tf.int32,
        )

        # mask is past the classify token (i.e. make those results extremely negative)
        mask = tf.expand_dims(
            1.0
            - tf.sequence_mask(
                pool_idx, maxlen=tf.shape(representation)[1], dtype=tf.float32
            ),
            -1,
        )
        pool = tf.reduce_max(representation + mask * -1e9, 1)
        clf_h = pool
        clf_h = tf.reshape(
            clf_h, shape=tf.concat((initial_shape[:-2], [config.n_filter]), 0)
        )

        # note that, due to convolution and pooling, the dimensionality of the features is much smaller than in the
        # transformer base models

        lengths = lengths_from_eos_idx(eos_idx=pool_idx, max_length=sequence_length)
        return {
            "features": clf_h,  # [batch_size, n_embed] for classify, [batch_size, 1, n_embed] for comparison, etc.
            "sequence_features": seq_feats,  # [batch_size, seq_len, n_embed]
            "eos_idx": pool_idx,  # [batch_size]
            "lengths": lengths
        }

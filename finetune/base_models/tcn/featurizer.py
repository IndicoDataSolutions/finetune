import tensorflow as tf

from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.gpt.featurizer import dropout, embed


class TemporalBlock:
    def __init__(self, n_filters, kernel_size, dilation_rate, rate, scope):
        self.scope = scope
        self.rate = rate
        self.n_filters = n_filters
        self.conv1 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu,
            dilation_rate=dilation_rate,
            kernel_initializer=tf.compat.v1.initializers.glorot_normal,
            name="conv1",
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.initializers.glorot_normal,
            name="conv2",
        )
        self.downsample = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=1, padding="same")

    def __call__(self, X):
        with tf.compat.v1.variable_scope(self.scope):
            conv1_out = self.conv1(X)
            conv1_dropout = tf.nn.dropout(conv1_out, rate=self.rate)
            conv2_out = self.conv2(conv1_dropout)
            conv2_dropout = tf.nn.dropout(conv2_out, rate=self.rate)

        # residuals
        if X.get_shape()[-1] == self.n_filters:
            output = conv2_dropout + X
        else:
            downsampled_out = self.downsample(X)
            output = conv2_dropout + downsampled_out

        return output


def tcn_featurizer(
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
    initial_shape = tf.shape(input=X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))
    sequence_length = tf.shape(input=X)[1]
    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        embed_weights = tf.compat.v1.get_variable(
            name="we",
            shape=[encoder.vocab_size + config.max_length, config.n_embed_featurizer],
            initializer=tf.compat.v1.random_normal_initializer(stddev=config.weight_stddev),
        )

        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)
            
        h = tf.gather(embed_weights, X)

        # keep track of the classify token
        clf_token = encoder["_classify_"]

        with tf.compat.v1.variable_scope("tcn_stack"):
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

        # mask out the values past the classify token before performing pooling
        pool_idx = tf.cast(
            tf.argmax(input=tf.cast(tf.equal(X, clf_token), tf.float32), axis=1),
            tf.int32,
        )

        # mask is past the classify token (i.e. make those results extremely negative)
        mask = tf.expand_dims(
            1.0
            - tf.sequence_mask(
                pool_idx, maxlen=tf.shape(input=representation)[1], dtype=tf.float32
            ),
            -1,
        )
        pool = tf.reduce_max(input_tensor=representation + mask * -1e9, axis=1)
        clf_h = pool
        clf_h = tf.reshape(
            clf_h, shape=tf.concat((initial_shape[:-1], [config.n_filter]), 0)
        )

        # note that, due to convolution and pooling, the dimensionality of the features is much smaller than in the
        # transformer base models

        lengths = lengths_from_eos_idx(eos_idx=pool_idx, max_length=sequence_length)
        return {
            "embed_weights": embed_weights,
            "features": clf_h,  # [batch_size, n_embed] for classify, [batch_size, 1, n_embed] for comparison, etc.
            "sequence_features": seq_feats,  # [batch_size, seq_len, n_embed]
            "eos_idx": pool_idx,  # [batch_size]
            "lengths": lengths
        }

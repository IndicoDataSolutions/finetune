import tensorflow as tf

from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.gpt.featurizer import dropout, embed
from finetune.nn.add_auxiliary import add_auxiliary


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
            kernel_initializer=tf.initializers.glorot_normal,
            name="conv1",
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal,
            name="conv2",
        )
        self.downsample = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=1, padding="same")

    def __call__(self, X):
        with tf.variable_scope(self.scope):
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
    context=None,
    context_dim=None,
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
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-2:]), 0))

    with tf.variable_scope("model/featurizer", reuse=reuse):
        embed_weights = tf.get_variable(
            name="we",
            shape=[encoder.vocab_size + config.max_length, config.n_embed_featurizer],
            initializer=tf.random_normal_initializer(stddev=config.weight_stddev),
        )

        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        X = tf.reshape(X, [-1, config.max_length, 2])

        # we remove positional embeddings from the model
        h = embed(X[:, :, :1], embed_weights)

        # keep track of the classify token
        clf_token = encoder["_classify_"]

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

        seq_feats = tf.reshape(representation, shape=[-1, config.max_length, config.n_filter])

        # mask out the values past the classify token before performing pooling
        pool_idx = tf.cast(
            tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1),
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

        if config.use_auxiliary_info:
            clf_h, seq_feats = add_auxiliary(
                context, context_dim, clf_h, seq_feats, config, train
            )

        # note that, due to convolution and pooling, the dimensionality of the features is much smaller than in the
        # transformer base models

        lengths = lengths_from_eos_idx(eos_idx=pool_idx, max_length=config.max_length)
        return {
            "embed_weights": embed_weights,
            "features": clf_h,  # [batch_size, n_embed] for classify, [batch_size, 1, n_embed] for comparison, etc.
            "sequence_features": seq_feats,  # [batch_size, seq_len, n_embed]
            "eos_idx": pool_idx,  # [batch_size]
            "lengths": lengths
        }

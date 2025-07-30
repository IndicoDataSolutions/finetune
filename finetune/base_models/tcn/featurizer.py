import tensorflow as tf

from finetune.nn.nn_utils import saver_ignore_scope


class TemporalBlock(tf.keras.layers.Layer):
    def __init__(
        self, *, n_filters, kernel_size, dilation_rate, dropout_rate, **kwargs
    ):
        super().__init__(**kwargs)
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
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        conv1_out = self.conv1(x)
        conv1_dropout = self.drop1(conv1_out)
        conv2_out = self.conv2(conv1_dropout)
        return self.drop2(conv2_out)


@saver_ignore_scope
class TemporalBlockWithResiduals(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        n_filters,
        kernel_size,
        dilation_rate,
        dropout_rate,
        name="block",
        **kwargs
    ):
        # It looks like a mistake that we are passing name to block and not to this. But we are ignoring
        # this scope and on the original the scope only wrapped the block and not the downsample. So we've gone
        # with this to achieve the same in keras. Weird but should be fine.
        super().__init__(**kwargs)
        self.block = TemporalBlock(
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
            name=name,
        )
        self.n_filters = n_filters

    def build(self, input_shape):
        if input_shape[-1] != self.n_filters:
            self.downsample = tf.keras.layers.Conv1D(
                filters=self.n_filters, kernel_size=1, padding="same"
            )
        else:
            self.downsample = None

    def call(self, x):
        output = self.block(x)
        # Odd that we are downsampling the residual stream and not the block output but that's
        # what the original did...
        if self.downsample is not None:
            return self.downsample(x) + output
        return x + output


class TCNStack(tf.keras.layers.Layer):
    def __init__(
        self, n_blocks, n_filters, kernel_size, dropout_rate, name="tcn_stack", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.blocks = [
            TemporalBlockWithResiduals(
                n_filters=n_filters,
                kernel_size=kernel_size,
                dilation_rate=2**layer_num,
                dropout_rate=dropout_rate,
                name="Temporal{}".format(layer_num),
            )
            for layer_num in range(n_blocks)
        ]

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TCNFeaturizer(tf.keras.layers.Layer):
    def __init__(self, encoder, config, **kwargs):
        super().__init__(**kwargs)
        self.tcn_stack = TCNStack(
            n_blocks=config.n_layer,
            n_filters=config.n_filter,
            kernel_size=config.kernel_size,
            dropout_rate=config.resid_p_drop,
        )
        self.weight_stddev = config.weight_stddev
        self.embedding_dropout = tf.keras.layers.Dropout(config.embed_p_drop)
        self.clf_token = encoder["_classify_"]
        self.max_length = config.max_length
        self.vocab_size = encoder.vocab_size
        self.n_embed = config.n_embed_featurizer

    def build(self, input_shape):
        self.embed_weights = self.add_weight(
            # Why do we have space for pos embeddings here? Can we safely drop this and let the saver handle the slicing?
            shape=[self.vocab_size + self.max_length, self.n_embed],
            initializer=tf.keras.initializers.RandomNormal(stddev=self.weight_stddev),
            trainable=True,
            name="we",
        )

    def call(self, tokens, context, sequence_lengths, training=True):
        embed_weights = self.embedding_dropout(self.embed_weights)
        h = tf.gather(embed_weights, tokens)
        seq_feats = self.tcn_stack(h)
        # Mask padding and max reduce.
        pool_idx = tf.cast(
            tf.argmax(
                input=tf.cast(tf.equal(tokens, self.clf_token), tf.float32), axis=1
            ),
            tf.int32,
        )
        mask = tf.expand_dims(
            1.0
            - tf.sequence_mask(pool_idx, maxlen=tf.shape(input=h)[1], dtype=tf.float32),
            -1,
        )
        pooled_out = tf.reduce_max(input_tensor=seq_feats + mask * -1e9, axis=1)

        return {
            "features": pooled_out,
            "sequence_features": seq_feats,
        }

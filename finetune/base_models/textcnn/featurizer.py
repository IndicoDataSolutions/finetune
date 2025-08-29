import tensorflow as tf

from finetune.nn.nn_utils import ExtraScope, saver_ignore_scope


@saver_ignore_scope
class Embedding(tf.keras.layers.Layer):
    def __init__(self, shape, stddev, embed_p_drop, weight_name, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.stddev = stddev
        self.embed_dropout = tf.keras.layers.Dropout(embed_p_drop)
        self.weight_name = weight_name

    def build(self, input_shape):
        self.embed_weights = self.add_weight(
            # Why do we have space for pos embeddings here? Can we safely drop this and let the saver handle the slicing?
            shape=self.shape,
            initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev),
            trainable=True,
            name=self.weight_name,
        )

    def call(self, tokens):
        return tf.gather(self.embed_dropout(self.embed_weights), tokens)


class TextCNNFeaturizer(tf.keras.layers.Layer):
    def __init__(self, encoder, config, **kwargs):
        super().__init__(**kwargs)
        self.convs = [
            tf.keras.layers.Conv1D(
                filters=config.num_filters_per_size,
                kernel_size=kernel_size,
                padding="same",
                activation=tf.nn.relu,
                name=f"conv{i}",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
            )
            for i, kernel_size in enumerate(config.kernel_sizes)
        ]
        # Why do we have space for pos embeddings here? Can we safely drop this and let the saver handle the slicing?
        if "roberta" in config.base_model_path:
            embed_name = "word_embeddings"
            embed_extra_scopes = ["bert", "embeddings"]
            self.clf_token = encoder.delimiter_token
        else:
            embed_name = "we"
            embed_extra_scopes = []
            self.clf_token = encoder["_classify_"]

        embedding = Embedding(
            shape=[encoder.vocab_size, config.n_embed],
            stddev=config.weight_stddev,
            embed_p_drop=config.embed_p_drop,
            weight_name=embed_name,
        )
        for extra_scope in reversed(embed_extra_scopes):
            embedding = ExtraScope(embedding, extra_scope)
        self.embedding = embedding

    def call(self, tokens, context, sequence_lengths):
        h = self.embedding(tokens)
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

        pool_layers = []
        conv_layers = []
        for conv_layer in self.convs:
            conv = conv_layer(h)
            conv_layers.append(conv)
            pool = tf.reduce_max(input_tensor=conv + mask * -1e9, axis=1)
            pool_layers.append(pool)

        # Concat the output of the convolutional layers for use in sequence embedding
        conv_seq = tf.concat(conv_layers, axis=2)

        # concat all reduced features
        pooled_out = tf.concat(pool_layers, axis=1)
        return {
            "features": pooled_out,
            "sequence_features": conv_seq,
        }

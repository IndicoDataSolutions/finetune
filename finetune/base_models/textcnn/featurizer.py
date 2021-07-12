import tensorflow as tf

from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.gpt.featurizer import dropout, embed


def textcnn_featurizer(X, encoder, config, train=False, reuse=None, **kwargs):
    """
    The transformer element of the finetuning model. Maps from tokens ids to a dense, embedding of the sequence.

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
        # Name of embed weights variable depends on tokenizer/base model weights file
        if "roberta" in config.base_model_path:
            embed_weights_name = "bert/embeddings/word_embeddings"
            clf_token = encoder.delimiter_token
        else:
            embed_weights_name = "we"
            clf_token = encoder["_classify_"]

        embed_weights = tf.compat.v1.get_variable(
            name=embed_weights_name,
            shape=[encoder.vocab_size, config.n_embed_featurizer],
            initializer=tf.compat.v1.random_normal_initializer(
                stddev=config.weight_stddev
            ),
        )

        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        h = tf.gather(embed_weights, X)

        # Keep track of the classify token. Token depends on tokenizer / base model weights file
        # if "roberta" in config.base_model_path:
        #     clf_token = encoder.delimiter_token
        # else:
        #     clf_token = encoder["_classify_"]

        # mask out the values past the classify token before performing pooling
        pool_idx = tf.cast(
            tf.argmax(input=tf.cast(tf.equal(X, clf_token), tf.float32), axis=1),
            tf.int32,
        )
        # mask is past the classify token (i.e. make those results extremely negative)
        mask = tf.expand_dims(
            1.0
            - tf.sequence_mask(pool_idx, maxlen=tf.shape(input=h)[1], dtype=tf.float32),
            -1,
        )

        # Convolutional Layer (this is all the same layer, just different filter sizes)
        pool_layers = []
        conv_layers = []
        for i, kernel_size in enumerate(config.kernel_sizes):
            conv = tf.compat.v1.layers.conv1d(
                inputs=h,
                filters=config.num_filters_per_size,
                kernel_size=kernel_size,
                padding="same",
                activation=tf.nn.relu,
                name="conv" + str(i),
                kernel_initializer=tf.compat.v1.initializers.glorot_normal,
            )
            conv_layers.append(conv)
            pool = tf.reduce_max(input_tensor=conv + mask * -1e9, axis=1)
            pool_layers.append(pool)

        # Concat the output of the convolutional layers for use in sequence embedding
        conv_seq = tf.concat(conv_layers, axis=2)
        seq_feats = tf.reshape(conv_seq, shape=[-1, sequence_length, config.n_embed])

        # Concatenate the univariate vectors as features for classification
        clf_h = tf.concat(pool_layers, axis=1)
        clf_h = tf.reshape(
            clf_h, shape=tf.concat((initial_shape[:-1], [config.n_embed]), 0)
        )

        # note that, due to convolution and pooling, the dimensionality of the features is much smaller than in the
        # transformer base models
        lengths = lengths_from_eos_idx(eos_idx=pool_idx, max_length=sequence_length)

        return {
            "embed_weights": embed_weights,
            "features": clf_h,  # [batch_size, n_embed] for classify, [batch_size, 1, n_embed] for comparison, etc.
            "sequence_features": seq_feats,  # [batch_size, seq_len, n_embed]
            "eos_idx": pool_idx,  # [batch_size]
            "lengths": lengths,
        }

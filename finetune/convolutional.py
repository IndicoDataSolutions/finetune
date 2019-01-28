import tensorflow as tf

from finetune.transformer import dropout, embed, block, attn, norm
from finetune.utils import shape_list


def embed_no_timing(X, we):
    return tf.gather(we, X[:, :, 0])

def gated_linear_unit(X, kernel_width, layer_name, use_fp16, residual=None):
    # layer_input shape = #batch, seq, embed_dim or batch, channels, seq, embed_dim

    with tf.variable_scope(layer_name):

        # Pad kernel_width (word_wise) - 1 to stop future viewing.
        left_pad = kernel_width - 1
        paddings = [[0, 0], [left_pad, 0], [0, 0]]
        padded_input = tf.pad(X, paddings, "CONSTANT")

        nx = shape_list(X)[-1]
        W = tf.get_variable(name="W", shape=[kernel_width, nx, nx])
        b = tf.get_variable(name="B", shape=[nx])
        W_gate = tf.get_variable(name="W_gate", shape=[kernel_width, nx, nx])
        b_gate = tf.get_variable(name="B_gate", shape=[nx])

        if use_fp16:
            W = tf.cast(W, tf.float16)
            b = tf.cast(b, tf.float16)
            W_gate = tf.cast(W_gate, tf.float16)
            b_gate = tf.cast(b_gate, tf.float16)

        conv = tf.nn.conv1d(
            padded_input,
            W,
            stride=1,
            padding="VALID",
            name="conv"
        )
        conv = tf.nn.bias_add(conv, b)

        # Second gating sigmoid layer

        conv_gate = tf.nn.conv1d(
            padded_input,
            W_gate,
            stride=1,
            padding="VALID",
            name="conv"
        )
        conv_gate = tf.nn.bias_add(conv_gate, b_gate)

        # residuals
        if residual is not None:
            conv = tf.add(conv, residual)
            conv_gate = tf.add(conv_gate, residual)

        h = tf.multiply(conv, tf.sigmoid(conv_gate, name="sig"))

    return h

def block(X, kernel_width, block_name, use_fp16):
    with tf.variable_scope(block_name):
        h0 = gated_linear_unit(X, kernel_width, "0",use_fp16)
        h1 = gated_linear_unit(h0, kernel_width, "1",use_fp16)
        h2 = gated_linear_unit(h1, kernel_width, "2",use_fp16)
        h3 = gated_linear_unit(h2, kernel_width, "3",use_fp16)
        h4 = gated_linear_unit(h3, kernel_width, "4",use_fp16, h0)
    return h4




def featurizer(X, encoder, config, train=False, reuse=None):
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
    initial_shape = [a or -1 for a in X.get_shape().as_list()]
    X = tf.reshape(X, shape=[-1] + initial_shape[-2:])

    with tf.variable_scope('model/featurizer', reuse=reuse):
        embed_weights = tf.get_variable("we", [encoder.vocab_size + config.max_length, config.n_embed],
                                        initializer=tf.random_normal_initializer(stddev=config.weight_stddev))
        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)
        X = tf.reshape(X, [-1, config.max_length, 2])

        if config.use_timing:
            h = embed(X, embed_weights)
        else:
            h = embed_no_timing(X, embed_weights)

        if config.use_fp16:
            h = tf.cast(h, tf.float16)

        for layer in range(config.n_layer):
            if (config.n_layer - layer) == config.num_layers_trained and config.num_layers_trained != config.n_layer:
                h = tf.stop_gradient(h)
            h = block(h, block_name='block%d_' % layer, kernel_width=config.kernel_width, use_fp16=config.use_fp16)

        if config.use_fp16:
            h = tf.cast(h, tf.float32)

        # Use hidden state at classifier token as input to final proj. + softmax
#        clf_h = tf.reshape(h, [-1, config.n_embed])  # [batch * seq_len, embed]
        clf_token = encoder['_classify_']
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        pool_mask  = tf.expand_dims(tf.sequence_mask(lengths=pool_idx, maxlen=config.max_length, dtype=tf.float32), -1)
        
#        clf_h = tf.reduce_sum(h * pool_mask, axis=1) / tf.reduce_sum(pool_mask, axis=1)
        clf_h = tf.reduce_max(h * pool_mask, axis=1)
        clf_h = tf.reshape(clf_h, shape=initial_shape[: -2] + [config.n_embed])
        seq_feats = tf.reshape(h, shape=initial_shape[:-1] + [config.n_embed])

        return {
            'embed_weights': embed_weights,
            'features': clf_h,
            'sequence_features': seq_feats
        }

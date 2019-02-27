import tensorflow as tf

from finetune.base_models.gpt.featurizer import dropout, embed, block, norm
from finetune.util.shapes import shape_list

from finetune.transformer import dropout, embed, block, attn, norm, conv1d
from finetune.optimizers.recompute_grads import recompute_grad
from finetune.utils import shape_list


def embed_no_timing(X, we):
    return tf.gather(we, X[:, :, 0])

def gated_linear_unit(X, kernel_width, layer_name, use_fp16, training, backwards=False, mask=None):
    # layer_input shape = #batch, seq, embed_dim or batch, channels, seq, embed_dim

    with tf.variable_scope(layer_name, reuse=backwards):

        # Pad kernel_width (word_wise) - 1 to stop future viewing.
        left_pad = kernel_width - 1
        if backwards:
            paddings = [[0, 0], [0, left_pad], [0, 0]]
        else:
            paddings = [[0, 0], [left_pad, 0], [0, 0]]
            
        if mask is not None:
            X *= mask
        padded_input = tf.pad(X, paddings, "CONSTANT")

        nx = shape_list(X)[-1]
        W = tf.get_variable(name="W", shape=[kernel_width, nx, nx], initializer=tf.initializers.random_normal(stddev=0.00001))
        b = tf.get_variable(name="B", shape=[nx], initializer=tf.initializers.random_normal(stddev=0.001))
        W_gate = tf.get_variable(name="W_gate", shape=[kernel_width, nx, nx], initializer=tf.initializers.random_normal(stddev=0.00001))
        b_gate = tf.get_variable(name="B_gate", shape=[nx], initializer=tf.initializers.random_normal(stddev=0.001))

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
        conv = tf.nn.relu(norm(conv, "norm", fp16=use_fp16))
        # Second gating sigmoid layer

        if mask is not None:
            conv *=mask
            
        conv = tf.pad(conv, paddings, "CONSTANT")
        conv_gate = tf.nn.conv1d(
            conv,
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

def block(X, kernel_width, block_name, use_fp16, training, pdrop, backwards=False, seq_lens=None):
    with tf.variable_scope(block_name, reuse=backwards):
        if seq_lens is not None:
            mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_lens, maxlen=tf.shape(X)[1], dtype=tf.float32), axis=-1), X.dtype)
        else:
            mask = None

        h0 = gated_linear_unit(X, kernel_width, "0",use_fp16, training, backwards, mask)
        h0 = attention_layer(h0, backwards, seq_lens, block_name + "_1")
        h0 = tf.nn.relu(h0)
                          
#        h1 = gated_linear_unit(h0, kernel_width, "1",use_fp16, training, backwards, mask)
#        h1 = tf.nn.relu(h1)
        h1 = h0
        
        h2 = gated_linear_unit(h1, kernel_width, "2",use_fp16, training, backwards, mask)
        h2 = attention_layer(h2, backwards, seq_lens, block_name + "_2")
        h2 = tf.nn.relu(h2)
        
        h4 = gated_linear_unit(h2, kernel_width, "4", use_fp16, training, backwards, mask)
    return h4


def attention_layer(X, backwards, seq_lens, layer):
    # X = batch, seq, features
    weight_orig = weight = X[:,:,0]
    dtype = weight.dtype
    batch, seq, feats = shape_list(X)
    weight = tf.ones([batch, seq, seq], dtype=dtype) * tf.expand_dims(weight, axis=1)
    if backwards:
        b = tf.matrix_band_part(tf.ones([seq, seq]), 0, -1)
    else:    
        b = tf.matrix_band_part(tf.ones([seq, seq]), -1, 0)
    if backwards and seq_lens is not None:
        b = tf.expand_dims(tf.sequence_mask(seq_lens, maxlen=tf.shape(X)[1], dtype=tf.float32), 1) * b


        
    b = tf.reshape(b, [1, seq, seq])
    if dtype == tf.float16:
        b = tf.cast(b, tf.float16)
    weight = weight * b + (-1e4 if dtype == tf.float16 else -1e9) * (1 - b)
    weight = tf.nn.softmax(weight, -1) * b
#    weight = tf.Print(weight, [weight[0,2,:]])

    tf.summary.image("attns_{}_at_{}".format("f" if not backwards else "b", layer), tf.expand_dims(tf.pow(weight, 0.2), -1))

    attn_size = feats // 3 - 1
    attention_bit = X[:,:,1: attn_size + 1]
    other_bit = X[:,:, attn_size + 1 :]
    out = tf.concat([tf.expand_dims(weight_orig, -1), tf.matmul(weight, attention_bit), other_bit], axis=-1)
    return out


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
        encoder._lazy_init()
        clf_token = encoder['_classify_']
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        embed_weights_orig = embed_weights = tf.get_variable("we", [encoder.vocab_size + config.max_length, config.n_embed],
                                        initializer=tf.random_normal_initializer(stddev=config.weight_stddev))
        if config.use_fp16:
            embed_weights = tf.cast(embed_weights, tf.float16)
            
        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)
            
        X = tf.reshape(X, [-1, config.max_length, 2])
        embed_size = config.n_embed

        if config.use_timing:
            h = embed(X, embed_weights)
        else:
            h = embed_no_timing(X, embed_weights)

        fe = tf.get_variable(name="forward_embed", shape=[1, 1, embed_size])
        be = tf.get_variable(name="backwards_embed", shape=[1, 1, embed_size])
        f_b_combine_w = tf.get_variable(name="f_b_combine", shape=[1, embed_size * 2, embed_size], initializer=tf.initializers.random_normal(stddev=0.00001))
        if config.use_fp16:
            fe = tf.cast(fe, tf.float16)
            be = tf.cast(be, tf.float16)
            f_b_combine_w = tf.cast(f_b_combine_w, tf.float16)

        h_back = h + be
        h += fe 
        for layer in range(config.n_layer):
            if (config.n_layer - layer) == config.num_layers_trained and config.num_layers_trained != config.n_layer:
                h = tf.stop_gradient(h)
            h = block(h, block_name='block%d_' % layer, kernel_width=config.kernel_width, use_fp16=config.use_fp16, training=train, pdrop=config.resid_p_drop)
            h_back = block(h_back, block_name='block%d_' % layer, kernel_width=config.kernel_width, use_fp16=config.use_fp16, training=train, pdrop=config.resid_p_drop, backwards=True, seq_lens=pool_idx)

        h = tf.concat((h, tf.concat((h_back[:, 2:], tf.zeros_like(h_back[:,:2])), axis=1)), axis=-1)
        embed_size = shape_list(h_back)[-1]

        h = tf.nn.conv1d(
                        h,
                        f_b_combine_w,
                        stride=1,
                        padding="VALID",
                        name="conv"
        )
        
        if config.use_fp16:
            h = tf.cast(h, tf.float32)

        mask = tf.expand_dims(tf.sequence_mask(pool_idx, maxlen=tf.shape(h)[1], dtype=tf.float32), -1)
        h = tf.Print(h, (tf.shape(h), tf.shape(mask)))
        max_pooled = tf.reduce_max(h + (1.0 -  mask) * -1e9, 1)
        mean_pool = tf.reduce_sum(h * mask, 1) / (tf.reduce_sum(mask) + 1e-9)
        clf_h = tf.concat((max_pooled, mean_pool), axis=1)
        
        clf_h = tf.reshape(clf_h, shape=initial_shape[: -2] + [config.n_embed * 2])
        seq_feats = tf.reshape(h, shape=initial_shape[:-1] + [config.n_embed * 2])

        return {
            'embed_weights': embed_weights_orig,
            'features': clf_h,
            'sequence_features': seq_feats,
            'pool_idx': pool_idx
        }

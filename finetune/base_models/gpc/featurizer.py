import tensorflow as tf
import math
from finetune.base_models.gpt.featurizer import dropout, embed, block, norm
from finetune.util.shapes import shape_list

from finetune.optimizers.recompute_grads import recompute_grad
import functools
from tensorflow.python.framework import function

def embed_no_timing(X, we):
    return tf.gather(we, X[:, :, 0])


def _swish_shape(op):
    """Shape helper function for swish and _swish_grad function below."""
    return [op.inputs[0].shape]


@function.Defun(shape_func=_swish_shape, func_name="swish_grad", noinline=True)
def _swish_grad(features, grad):
    sigmoid_features = tf.nn.sigmoid(features)
    activation_grad = (sigmoid_features * (1.0 + features * (1.0 - sigmoid_features)))
    return grad * activation_grad


@function.Defun(grad_func=_swish_grad, shape_func=_swish_shape, func_name="swish", noinline=True)
def swish(features):
    return features * tf.nn.sigmoid(features)


def mock_separable_conv(x, depthwise_filter, pointwise_filter):
    effective_filters = tf.einsum('hcm,cmn->hcn', depthwise_filter, pointwise_filter)
    return tf.nn.conv1d(x, effective_filters, padding="VALID", stride=1 )


def super_duper_separable_conv1d(x, depthwise_filter, pointwise_filter):
    # depthwise_filter [kernel_width, 1, channel_mult]
    # pointwise filter [1, channel_mult, nx]
    # x [batch, seq, feats]

    depthwise_out = tf.nn.conv2d(
        tf.expand_dims(x, 3),  # [batch, seq, feats, 1]
        tf.expand_dims(depthwise_filter, 2),  # [kernel_width, 1, 1, feat_out]
        strides=[1, 1, 1, 1],
        padding="VALID"
    )
    batch, seq, feat, channel_mult = shape_list(depthwise_out)

    depthwise_out = tf.reshape(depthwise_out, [batch, seq, feat * channel_mult])
    pointwise_out = tf.nn.conv1d(depthwise_out, pointwise_filter, stride=1, padding='VALID')
    return pointwise_out


def separable_conv1d(x, depthwise_filter, pointwise_filter):
    x = tf.expand_dims(x, 1)
    depthwise_filter = tf.expand_dims(depthwise_filter, 0)
    pointwise_filter = tf.expand_dims(pointwise_filter, 0)
    out = tf.nn.separable_conv2d(
        x,
        depthwise_filter,
        pointwise_filter,
        [1, 1, 1, 1],
        "VALID"
    )
    return tf.squeeze(out, 1)


def time_to_batch(value, dilation, pad_with=0):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]], constant_values=pad_with)
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])

def dilated_causal_max_pool(value, kernel_size, dilation, dim=1):
    pool_op = lambda x: causalmax(x, dim=1, pool_len=kernel_size)
    with tf.name_scope("causal_pool"):
        if dilation > 1:
            transformed = time_to_batch(value, dilation, pad_with=-1e4)
            pooled = pool_op(transformed)
            restored = batch_to_time(pooled, dilation)
        else:
            restored = pool_op(value)
        return restored

def cascaded_pool(value, kernel_size, dim=1, pool_len=None):
    shape = shape_list(value)
    full_pool_len = pool_len or shape[dim]
    intermediate_vals = []
    for i in range(math.ceil(math.log(full_pool_len, kernel_size))):
        value = dilated_causal_max_pool(value, kernel_size=kernel_size, dilation=kernel_size ** i, dim=dim)
        w = tf.get_variable("weighted_mean_max_pool_{}".format(i), shape=[shape[-1]], dtype=tf.float32)
        if w.dtype != value.dtype:
            w = tf.cast(w, value.dtype)
        intermediate_vals.append(value * w)
    return tf.reduce_mean(intermediate_vals, 0)
    
def causal_conv(value, filter_, dilation, name='causal_conv'):
    if type(filter_) == tuple:
        conv_op = lambda x: mock_separable_conv(x, *filter_)
    else:
        conv_op = lambda x: tf.nn.conv1d(x, filter_, stride=1, padding='VALID')

    with tf.name_scope(name):
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = conv_op(transformed)
            restored = batch_to_time(conv, dilation)
        else:
            restored = conv_op(value)
        return restored


def semi_separable_conv_block(X, kernel_width, layer_name, use_fp16, training, mask=None, dilation=1, separation=8):
    with tf.variable_scope("sep_" + layer_name):

        left_pad = (kernel_width - 1) * dilation
        paddings = [[0, 0], [left_pad, 0], [0, 0]]

        padded_input = tf.pad(X, paddings, "CONSTANT")

        nx = shape_list(X)[-1]
        channels_per_slice = nx // separation
        b = tf.get_variable(name="B", shape=[nx], initializer=tf.zeros_initializer())
        if use_fp16:
            b = tf.cast(b, tf.float16)

        s = tf.concat(tf.split(padded_input, separation, axis=-1), 0)
        W = tf.get_variable(name="W", shape=[kernel_width, channels_per_slice, channels_per_slice],
                            initializer=tf.initializers.random_normal(stddev=0.00001))
        if use_fp16:
            W = tf.cast(W, tf.float16)
        conv = causal_conv(s, W, dilation)
        conv = tf.concat(tf.split(conv, separation, axis=0), -1)

        conv = tf.nn.bias_add(conv, b)
        out = conv

    return out

def cummaxv1(x, dim):
    def align_to_0(tensor):
        ranks = list(range(len(shape_list(tensor))))
        if dim != 0:
            ranks[0], ranks[dim] = ranks[dim], ranks[0]
            return tf.transpose(tensor, ranks)
        else:
            return tensor
    return align_to_0(tf.scan(lambda a, b: tf.maximum(a, b), align_to_0(x)))

def causalmax(x, dim, pool_len=None):
    kernel_size = [1,1,1]
    pool_len = pool_len or shape_list(x)[dim]
    kernel_size[dim] = pool_len
    padding = [[0, 0], [0, 0], [0, 0]]
    padding[dim] = [pool_len - 1, 0]
    padded_x = tf.pad(x, padding, "CONSTANT", constant_values=-1e4)

    kernel_size.insert(2, 1)

    return tf.squeeze(
        tf.nn.max_pool(
            value=tf.expand_dims(padded_x, 2),
            ksize=kernel_size,
            strides=[1, 1, 1, 1],
            padding="VALID"
        ),
        axis=2
    )

def cummax(x, dim):
    return causalmax(x, dim, pool_len=None)



def cumulative_state_net(X, name, use_fp16, pool_idx):
    outputs = []
    nx = shape_list(X)[-1]
    output_sz = nx // 4
    with tf.variable_scope(name):
        for kernel in [1, 2, 4, 8]:
            outputs.append(normal_1d_conv_block(X, kernel, str(kernel), use_fp16, output_dim=output_sz))
    outputs_concat = tf.nn.relu(tf.concat(outputs, -1))
    cum_pooled = cascaded_pool(outputs_concat, kernel_size=8, pool_len=512) #cummax(outputs_concat, 1)
    outputs_cum_pooled = tf.concat([cum_pooled, X], -1)
    print(pool_idx)
    feats = tf.gather_nd(cum_pooled, pool_idx)
    feats_weight = tf.get_variable(name="featweights", shape=[nx], initializer=tf.ones_initializer())
    if use_fp16:
        feats_weight = tf.cast(feats_weight, tf.float16)
    feats  = tf.stop_gradient(feats) * feats_weight
    return normal_1d_conv_block(outputs_cum_pooled, 1, "output_reproject", use_fp16, output_dim=nx), feats


def normal_1d_conv_block(X, kernel_width, layer_name, use_fp16, dilation=1, layer_num=1, output_dim=None):
    # layer_input shape = #batch, seq, embed_dim or batch, channels, seq, embed_dim
    with tf.variable_scope(layer_name):
        # Pad kernel_width (word_wise) - 1 to stop future viewing.
        left_pad = (kernel_width - 1) * dilation
        paddings = [[0, 0], [left_pad, 0], [0, 0]]

        padded_input = tf.pad(X, paddings, "CONSTANT")

        nx = shape_list(X)[-1]
        if output_dim is None:
            output_dim = nx
        W = tf.get_variable(name="W", shape=[kernel_width, nx, output_dim], initializer=tf.initializers.glorot_normal())
        b = tf.get_variable(name="B", shape=[output_dim], initializer=tf.initializers.constant(0.0))

        if use_fp16:
            W = tf.cast(W, tf.float16)
            b = tf.cast(b, tf.float16)

        conv = causal_conv(padded_input, W, dilation)
        conv = tf.nn.bias_add(conv, b)

        out = conv
    return out


def block(X, block_name, use_fp16, layer_num=None, pool_idx=None):
    with tf.variable_scope(block_name):
        c, feats = cumulative_state_net(X, "cumulative_state_net", use_fp16, pool_idx)
        h0 = normal_1d_conv_block(c, 2, "0", use_fp16, dilation=1, layer_num=layer_num * 2 + 1)
        h0 = tf.nn.relu(h0)
        h1 = normal_1d_conv_block(h0, 2, "1", use_fp16, dilation=1)
        h1 = tf.nn.relu(h1)
        h2 = normal_1d_conv_block(h1, 2, "2", use_fp16,dilation=1)
        h2 = tf.nn.relu(h2)
        h3 = normal_1d_conv_block(h2, 2, "3", use_fp16, dilation=1)
        return tf.nn.relu(norm(h3 + X, "norm", fp16=use_fp16, e=1e-2)), feats


def attention_layer(X, backwards, seq_lens, layer):
    # X = batch, seq, features
    weight_orig = weight = X[:, :, 0]
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

    tf.summary.image("attns_{}_at_{}".format("f" if not backwards else "b", layer),
                     tf.expand_dims(tf.pow(weight, 0.2), -1))

    attn_size = feats // 3 - 1
    attention_bit = X[:, :, 1: attn_size + 1]
    other_bit = X[:, :, attn_size + 1:]
    print(weight_orig)
    exit()
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
        embed_weights = tf.get_variable("we", [encoder.vocab_size + config.max_length, config.n_embed],
                                        initializer=tf.random_normal_initializer(stddev=config.weight_stddev))
        if config.use_fp16:
            embed_weights = tf.cast(embed_weights, tf.float16)

        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        X = tf.reshape(X, [-1, config.max_length, 2])

        if config.use_timing:
            h = embed(X, embed_weights)
        else:
            h = embed_no_timing(X, embed_weights)
        feats = []
        for layer in range(config.n_layer):
            with tf.variable_scope('h%d_' % layer):
                if (
                        (config.n_layer - layer) == config.num_layers_trained and
                        config.num_layers_trained != config.n_layer
                ):
                    h = tf.stop_gradient(h)

                block_fn_fwd = functools.partial(block, block_name='block%d_' % layer, use_fp16=config.use_fp16,
                                                 layer_num=layer + 1, pool_idx=tf.stack([tf.range(shape_list(X)[0]), pool_idx], 1))
                if config.low_memory_mode and train:
                    block_fn_fwd = recompute_grad(block_fn_fwd, use_entire_scope=True)
                h, feats_i = block_fn_fwd(h)
                feats.append(feats_i)
                if config.n_layer == (layer + 1):
                    h = normal_1d_conv_block(h, 1, "output", config.use_fp16, dilation=1, layer_num=(config.n_layer + 1) * 2 + 1)

        mask = tf.expand_dims(tf.sequence_mask(pool_idx, maxlen=tf.shape(h)[1], dtype=h.dtype), -1)

#        max_pooled = tf.reduce_max(h + (1.0 - mask) * -1e9, 1)
#        mean_pool = tf.reduce_sum(h * mask, 1) / (tf.reduce_sum(mask) + 1e-9)
#        clf_h = tf.concat((max_pooled, mean_pool), axis=1)

        clf_h = tf.reshape(tf.reduce_sum(feats, 0), shape=initial_shape[: -2] + [config.n_embed])
        seq_feats = tf.reshape(h, shape=initial_shape[:-1] + [config.n_embed])

        return {
            'embed_weights': embed_weights,
            'features': tf.cast(clf_h, tf.float32),
            'sequence_features': seq_feats,
            'pool_idx': pool_idx
        }

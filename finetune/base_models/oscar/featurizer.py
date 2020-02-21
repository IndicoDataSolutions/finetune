import tensorflow as tf
#import tensorflow_addons as tfa
import numpy as np

from finetune.base_models.gpt.featurizer import dropout, embed, split_heads, merge_heads
from finetune.util.shapes import shape_list, lengths_from_eos_idx
from finetune.base_models.oscar.ra import recursive_agg, recursive_agg_tf
from finetune.optimizers.recompute_grads import recompute_grad

import functools

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
              (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
#TODO@    return tfa.activations.gelu(x)


def layer_norm(input_tensor):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1
    )


def cast_maybe(t, dtype):
    if t.dtype == dtype:
        return t
    return tf.cast(t, dtype=dtype)


def embed_no_timing(X, we):
    return tf.gather(we, X[:, :, 0])


def dual_linear(x, hidden_dim, output_dim, use_fp16):
    nx = shape_list(x)[-1]
    with tf.variable_scope("dual_linear"):
        w1 = tf.get_variable(name="W1", shape=[1, nx, hidden_dim], initializer=tf.initializers.glorot_normal())
        w2 = tf.get_variable(name="W2", shape=[1, hidden_dim, output_dim], initializer=tf.initializers.glorot_normal())
        if use_fp16:
            w1 = tf.cast(w1, tf.float16)
            w2 = tf.cast(w2, tf.float16)
    hidden = tf.nn.conv1d(x, w1, stride=1, padding='VALID')
    return tf.nn.conv1d(hidden, w2, stride=1, padding='VALID')


def pooling_through_time(*, x, kernel_size, pool_len, bidirectional, dual_linear_hidden, use_fp16, dim=1, use_fused_kernel=True):
    shape = shape_list(x)
    feats = shape[2]
    if use_fused_kernel:
        ra = recursive_agg
    else:
        ra = recursive_agg_tf

    if bidirectional:
        pool_len = pool_len // 2
        pool_input = tf.concat((x, tf.reverse(x, axis=[1])), axis=2) # concat forward and backwards together.
    else:
        pool_input = x
        
    aggregated = ra(pool_input, kernel_size, pool_len) # batch, seq, pooling_ops, feats

    if bidirectional:
        aggregated = tf.concat((aggregated[:, :, :, :feats], tf.reverse(aggregated[:, :, :, feats:], axis=[1])), axis=2)

    _, _, pool_ops, feats_pooled = shape_list(aggregated)
    
    weights = tf.nn.softmax(dual_linear(x, dual_linear_hidden, pool_ops * feats_pooled, use_fp16))
    weighted_over_time = tf.reduce_sum(aggregated * tf.reshape(weights, tf.shape(aggregated)), 2)
    
    return layer_norm(weighted_over_time + x)

def conv_stack(*, x, num_convs, filter_width, use_fp16, bidirectional):
    nx = shape_list(x)[-1]
    with tf.variable_scope("conv_stack"):
        inp = x
        for i in range(num_convs):
            if i != 0:
                inp = gelu(inp)
            inp = conv_1d_block(
                x=inp,
                kernel_width=filter_width,
                layer_name=str(i),
                use_fp16=use_fp16,
                output_dim=None,
                causal=not bidirectional
            )
    return layer_norm(inp + x)

def ffn(*, x, hidden_dim, use_fp16):
    nx = shape_list(x)[-1]
    with tf.variable_scope("ffn"):
        inp = x
        inp = conv_1d_block(
	    inp,
            kernel_width=1,
	    layer_name="1",
            use_fp16=use_fp16,
	    output_dim=hidden_dim,
            causal=False
        )
        inp = gelu(inp)
        inp = conv_1d_block(
	    inp,
            kernel_width=1,
	    layer_name="2",
            use_fp16=use_fp16,
            output_dim=nx,
            causal=False
        )
    return layer_norm(inp + x)


def conv_1d_block(x, kernel_width, layer_name, use_fp16, output_dim=None, causal=True):
    with tf.variable_scope(layer_name):
        if kernel_width > 1 and causal:
            left_pad = (kernel_width - 1) * dilation
            paddings = [[0, 0], [left_pad, 0], [0, 0]]
            padded_input = tf.pad(x, paddings, "CONSTANT")
        else:
            padded_input = x

        nx = shape_list(x)[-1]
        if output_dim is None:
            output_dim = nx

        w = tf.get_variable(name="W", shape=[kernel_width, nx, output_dim], initializer=tf.initializers.glorot_normal())
        b = tf.get_variable(name="B", shape=[output_dim], initializer=tf.initializers.constant(0.0))

        if use_fp16:
            W = tf.cast(W, tf.float16)
            b = tf.cast(b, tf.float16)
        conv = tf.nn.conv1d(padded_input, w, stride=1, padding='VALID' if causal else "SAME")
        conv = tf.nn.bias_add(conv, b)
    return conv


def block(
        x,
        num_convs,
        conv_filter_width,
        pool_filter_width,
        bidirectional,
        use_fp16,
        nominal_pool_length,
        dual_linear_hidden,
        ffn_hidden_dim,
        use_fused_kernel=True
):
    with tf.variable_scope("oscar_block"):
        x = conv_stack(
            x=x,
            num_convs=num_convs,
            filter_width=conv_filter_width,
            use_fp16=use_fp16,
            bidirectional=bidirectional
        )
        x = pooling_through_time(
            x=x,
            kernel_size=pool_filter_width,
            pool_len=nominal_pool_length,
            bidirectional=bidirectional,
            dual_linear_hidden=dual_linear_hidden,
            use_fp16=use_fp16,
            use_fused_kernel=True
        )
        return ffn(x=x, hidden_dim=ffn_hidden_dim, use_fp16=use_fp16)

def featurizer(X, encoder, config, train=False, reuse=None, encoder_state=None, context=None, context_dim=None, **kwargs):
    """
    The main element of the OSCAR model. Maps from tokens ids to a dense, embedding of the sequence.

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
    if len(initial_shape) != 3:
        X = tf.reshape(X, shape=[-1] + initial_shape[-2:])

    x_shape = tf.shape(X)
    with tf.variable_scope('model/featurizer', reuse=reuse):
        encoder._lazy_init()
        clf_token = encoder.end_token
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        if encoder_state is None:
            embed_weights = tf.get_variable("we", [encoder.vocab_size, config.n_embed],
                                            initializer=tf.random_normal_initializer(stddev=config.weight_stddev))
        else:
            embed_weights = encoder_state["embed_weights"]

        if config.oscar_use_fp16:
            embed_weights = tf.cast(embed_weights, tf.float16)

        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        X = tf.reshape(X, [-1, x_shape[1], 2])

        if config.oscar_use_timing:
            h = embed(X, embed_weights)
        else:
            h = embed_no_timing(X, embed_weights)

        for layer in range(config.n_layer):
            with tf.variable_scope('h%d_' % layer):
                if (
                        (config.n_layer - layer) == config.num_layers_trained and
                        config.num_layers_trained != config.n_layer
                ):
                    h = tf.stop_gradient(h)

                block_fn_fwd = functools.partial(
                    block,
                    num_convs=config.oscar_num_convs,
                    conv_filter_width=config.oscar_conv_filter_width,
                    pool_filter_width=config.oscar_pool_filter_width,
	            bidirectional=config.oscar_bidirectional,
                    use_fp16=config.oscar_use_fp16,
                    nominal_pool_length=config.oscar_nominal_pool_length,
                    dual_linear_hidden=config.oscar_dual_linear_hidden,
                    ffn_hidden_dim=config.oscar_ffn_hidden_dim,
                    use_fused_kernel=config.oscar_use_fused_kernel
                )

                if config.low_memory_mode and train:
                    block_fn_fwd = recompute_grad(block_fn_fwd, use_entire_scope=True)
                h = block_fn_fwd(h)

        mask = tf.expand_dims(tf.sequence_mask(pool_idx, maxlen=tf.shape(h)[1], dtype=h.dtype), -1)

        if config.oscar_feat_mode == "clf_tok":
            clf_h = tf.gather_nd(h, tf.stack([tf.range(shape_list(h)[0]), pool_idx], 1))
        elif config.oscar_feat_mode == "mean_tok":
            clf_h = tf.reduce_sum(h * mask, 1) / tf.reduce_sum(h)
        elif config.oscar_feat_mode == "max_tok":
            clf_h = tf.reduce_max(h - (1e5 * (1.0 - mask)), 1)
        else:
            raise ValueError("config.feat_mode should be one of clf_tok, mean_tok or max_tok")

        if len(initial_shape) != 3:
            seq_feats = tf.reshape(h, shape=initial_shape[:-1] + [config.n_embed])
        else:
            seq_feats = h

        return {
            'embed_weights': embed_weights,
            'features': cast_maybe(clf_h, tf.float32),
            'sequence_features': seq_feats,
            'eos_idx': pool_idx,
            'encoded_input': X[:, :tf.reduce_min(pool_idx), 0],
            'lengths': lengths_from_eos_idx(eos_idx=pool_idx, max_length=shape_list(X)[0])
        }

import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

from finetune.util.shapes import shape_list
from finetune.optimizers.recompute_grads import recompute_grad
from finetune.nn.activations import gelu
from finetune.base_models.gpt.featurizer import norm, dropout


def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, start + [nf])
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams, train=False):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_heads == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_heads), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v, attn_p_drop, train=False):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, attn_p_drop, train)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v, hparams.attn_p_drop, train=train)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        a = dropout(a, hparams.resid_p_drop, train=train)
        return a


def mlp(x, scope, n_state, *, hparams, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        h2 = dropout(h2, hparams.resid_p_drop, train=train)
        return h2


def block(x, *, past, hparams, train=False):
    nx = x.shape[-1].value
    a = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams, train=train)
    x = x + a
    m = mlp(norm(x, 'ln_2'), 'mlp', nx * 4, hparams=hparams, train=train)
    x = x + m
    return x


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_heads, sequence, hparams.n_embd // hparams.n_heads]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def embed(X, we):
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h


def gpt2_featurizer(X, encoder, config, train=False, reuse=None):
    initial_shape = tf.shape(X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-2:]), 0))

    with tf.variable_scope('model/featurizer', reuse=reuse):
        embed_weights = tf.get_variable(
            name="we",
            shape=[encoder.vocab_size + config.max_length, config.n_embed],
            initializer=tf.random_normal_initializer(stddev=config.weight_stddev)
        )
        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        X = tf.reshape(X, [-1, config.max_length, 2])
        h = embed(X, embed_weights)

        # Transformer
        pasts = [None] * config.n_layer
        for layer, past in enumerate(pasts):
            if (config.n_layer - layer) == config.num_layers_trained and config.num_layers_trained != config.n_layer:
                h = tf.stop_gradient(h)
                train_layer = False
            else:
                train_layer = train

            with tf.variable_scope('h%d' % layer):
                block_fn = functools.partial(block, past=past, hparams=config, train=train)
                if config.low_memory_mode and train_layer:
                    block_fn = recompute_grad(block_fn, use_entire_scope=True)
                h = block_fn(h)

        h = norm(h, 'ln_f')

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h, [-1, config.n_embed])  # [batch * seq_len, embed]
        clf_token = encoder['_classify_']
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * config.max_length + pool_idx)
        clf_h = tf.reshape(clf_h, shape=tf.concat((initial_shape[: -2], [config.n_embed]), 0))
        seq_feats = tf.reshape(h, shape=tf.concat((initial_shape[: -1], [config.n_embed]), 0))

        return {
            'embed_weights': embed_weights,
            'features': clf_h,
            'sequence_features': seq_feats,
            'pool_idx': pool_idx
        }

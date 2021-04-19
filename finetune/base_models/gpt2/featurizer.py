import functools

import tensorflow as tf

from finetune.util.shapes import shape_list, lengths_from_eos_idx
from finetune.optimizers.recompute_grads import recompute_grad
from finetune.nn.activations import gelu
from finetune.base_models.gpt.featurizer import norm, dropout, get_pos_values


def softmax(x, axis=-1):
    x = x - tf.reduce_max(input_tensor=x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(input_tensor=ex, axis=axis, keepdims=True)


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.compat.v1.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.compat.v1.get_variable(
            "w",
            [1, nx, nf],
            initializer=tf.compat.v1.random_normal_initializer(stddev=w_init_stdev),
        )
        b = tf.compat.v1.get_variable("b", [nf], initializer=tf.compat.v1.constant_initializer(0))
        c = tf.reshape(
            tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b,
            start + [nf],
        )
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams, train=False):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_heads == 0
    if past is not None:
        assert (
            past.shape.ndims == 5
        )  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(a=split_states(x, hparams.n_heads), perm=[0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(a=x, perm=[0, 2, 1, 3]))

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
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, attn_p_drop, train)
        a = tf.matmul(w, v)
        return a

    with tf.compat.v1.variable_scope(scope):
        c = conv1d(x, "c_attn", n_state * 3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v, hparams.attn_p_drop, train=train)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state)
        a = dropout(a, hparams.resid_p_drop, train=train)
        return a


def mlp(x, scope, n_state, *, hparams, train=False):
    with tf.compat.v1.variable_scope(scope):
        nx = x.shape[-1]
        h = gelu(conv1d(x, "c_fc", n_state))
        h2 = conv1d(h, "c_proj", nx)
        h2 = dropout(h2, hparams.resid_p_drop, train=train)
        return h2


def block(x, *, past, hparams, train=False):
    nx = x.shape[-1]
    a = attn(norm(x, "ln_1"), "attn", nx, past=past, hparams=hparams, train=train)
    x = x + a
    m = mlp(norm(x, "ln_2"), "mlp", nx * 4, hparams=hparams, train=train)
    x = x + m
    return x


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [
        batch_size,
        hparams.n_layer,
        2,
        hparams.n_heads,
        sequence,
        hparams.n_embd // hparams.n_heads,
    ]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value=value, name="value")
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    batch_size = tf.shape(input=tokens)[0]
    nsteps = tf.shape(input=tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def embed(X, we):
    e = tf.gather(we, X)
    h = tf.reduce_sum(input_tensor=e, axis=2)
    return h


def gpt2_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    **kwargs
):
    initial_shape = tf.shape(input=X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))
    X.set_shape([None, None])

    pos_values = get_pos_values(tf.shape(input=X)[1], encoder.vocab_size)
    X = tf.stack((X, tf.tile(pos_values, [tf.shape(input=X)[0], 1])), 2)

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        embed_weights = tf.compat.v1.get_variable(
            name="we",
            shape=[encoder.vocab_size + config.max_length, config.n_embed],
            initializer=tf.compat.v1.random_normal_initializer(stddev=config.weight_stddev),
        )
        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)
        h = embed(X, embed_weights)

        # Transformer
        pasts = [None] * config.n_layer
        for layer, past in enumerate(pasts):
            if (
                (config.n_layer - layer) == config.num_layers_trained
                and config.num_layers_trained != config.n_layer
            ):
                h = tf.stop_gradient(h)
                train_layer = False
            else:
                train_layer = train

            with tf.compat.v1.variable_scope("h%d" % layer):
                block_fn = functools.partial(
                    block, past=past, hparams=config, train=train
                )
                if config.low_memory_mode and train_layer:
                    block_fn = recompute_grad(block_fn, use_entire_scope=True)
                h = block_fn(h)

        h = norm(h, "ln_f")

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h, [-1, config.n_embed])  # [batch * seq_len, embed]
        clf_token = encoder["_classify_"]
        pool_idx = tf.cast(
            tf.argmax(input=tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), axis=1), tf.int32
        )
        batch, length, _ = shape_list(X)
        clf_h = tf.gather(
            clf_h,
            tf.range(batch, dtype=tf.int32) * length + pool_idx,
        )
        clf_h = tf.reshape(
            clf_h, shape=tf.concat((initial_shape[:-1], [config.n_embed]), 0)
        )
        seq_feats = tf.reshape(
            h, shape=tf.concat((initial_shape, [config.n_embed]), 0)
        )

        lengths = lengths_from_eos_idx(eos_idx=pool_idx, max_length=shape_list(X)[0])

        return {
            "embed_weights": embed_weights,
            "features": clf_h,
            "sequence_features": seq_feats,
            "eos_idx": pool_idx,
            "length": lengths
        }

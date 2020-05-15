import functools

import numpy as np
import tensorflow as tf

from finetune.optimizers.recompute_grads import recompute_grad
from finetune.util.shapes import shape_list, lengths_from_eos_idx
from finetune.nn.activations import act_fns
from finetune.nn.nn_utils import dropout, norm


def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.linalg.band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w * b + -1e9 * (1 - b)
    return w


def mask_pad(w, lengths):
    batch = shape_list(lengths)[0]
    maxlen = tf.cast(tf.reduce_max(input_tensor=lengths), tf.int32)
    seq_mask = tf.reshape(tf.sequence_mask(lengths, maxlen=maxlen), [batch, 1, 1, maxlen])
    b = tf.cast(seq_mask, tf.float32)
    w = w * b + -1e9 * (1 - b)
    return w


def explain_mask_attn_weights(w):
    # w is [batch, heads, n, n]
    # lengths is [batch]
    batch, _, _, n = shape_list(w)
    seq = n // 2
    main_mask = tf.linalg.band_part(tf.ones([seq, seq]), -1, 0)
    top = tf.expand_dims(
        tf.concat((main_mask, tf.zeros([seq, seq])), 1), 0
    )  # 1, seq, 2 * seq
    clf_to_clf_mask = tf.eye(seq)
    bottom = tf.expand_dims(
        tf.concat((main_mask, clf_to_clf_mask), 1), 0
    )  # 1, seq, 2 * seq
    m = tf.concat((top, bottom), 1)
    b = tf.reshape(m, [1, 1, n, n])
    w = w * b + -1e9 * (1 - b)
    return w


def attn_weights(q, k, v, scale=False, mask=True, explain=False, lengths=None):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w * tf.math.rsqrt(tf.cast(n_state, tf.float32))

    if mask:
        if explain:
            w = explain_mask_attn_weights(w)
        else:
            w = mask_attn_weights(w)
    elif lengths is not None:
        # at least mask pad tokens
        w = mask_pad(w, lengths=lengths)

    w = tf.nn.softmax(w)
    return w


def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)


def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


def split_heads(x, n, k=False):
    if k:
        return tf.transpose(a=split_states(x, n), perm=[0, 2, 3, 1])
    else:
        return tf.transpose(a=split_states(x, n), perm=[0, 2, 1, 3])


def merge_heads(x):
    return merge_states(tf.transpose(a=x, perm=[0, 2, 1, 3]))


def conv1d(
    x,
    scope,
    nf,
    rf,
    w_init=tf.compat.v1.random_normal_initializer(stddev=0.02),
    b_init=tf.compat.v1.constant_initializer(0),
    pad="VALID",
    train=False,
):
    with tf.compat.v1.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.compat.v1.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.compat.v1.get_variable("b", [nf], initializer=b_init)
        if rf == 1:  # faster 1x1 conv
            c = tf.reshape(
                tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b,
                shape_list(x)[:-1] + [nf],
            )
        else:  # was used to train LM
            c = tf.nn.conv1d(input=x, filters=w, stride=1, padding=pad) + b
        return c


def multihead_qkv(x, n_state, n_head, train, explain=False):
    c = conv1d(x, "c_attn", n_state * 3, 1, train=train)
    q, k, v = tf.split(c, 3, 2)
    q = split_heads(q, n_head)
    k = split_heads(k, n_head, k=True)
    v = split_heads(v, n_head)
    return q, k, v


def attn(
    x,
    scope,
    n_state,
    n_head,
    resid_pdrop,
    attn_pdrop,
    train=False,
    scale=False,
    mask=True,
    explain=False,
    lengths=None,
):
    assert n_state % n_head == 0
    with tf.compat.v1.variable_scope(scope):
        q, k, v = multihead_qkv(x, n_state, n_head, train, explain)
        w = attn_weights(q, k, v, scale=scale, mask=mask, explain=explain, lengths=lengths)
        w = dropout(w, attn_pdrop, train)
        a = tf.matmul(w, v)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a


def mlp(x, scope, n_state, act_fn, resid_pdrop, train=False):
    with tf.compat.v1.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[act_fn]
        h = act(conv1d(x, "c_fc", n_state, 1, train=train))
        h2 = conv1d(h, "c_proj", nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2


def create_initializer(initializer_range=0.001):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


def block(
    x,
    n_head,
    act_fn,
    resid_pdrop,
    attn_pdrop,
    scope,
    train=False,
    scale=False,
    explain=False,
):
    with tf.compat.v1.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(
            x,
            "attn",
            nx,
            n_head,
            resid_pdrop,
            attn_pdrop,
            train=train,
            scale=scale,
            explain=explain,
        )
        n = norm(x + a, "ln_1")
        m = mlp(n, "mlp", nx * 4, act_fn, resid_pdrop, train=train)
        h = norm(n + m, "ln_2")
        return h


def embed(X, we):
    return tf.reduce_sum(input_tensor=tf.gather(we, X), axis=2)


def add_explain_tokens(X, max_length, pool_idx):
    flat_x = tf.reshape(X[:, :, :1], [-1, 1])
    flat_pos = tf.minimum(
        X[:, :, 1:] + 1, max_length - 1
    )  # + 1 to offset for start token
    clf_tok = tf.gather(
        flat_x, tf.range(shape_list(X)[0], dtype=tf.int32) * max_length + pool_idx
    )
    clf_tok_x_seq = tf.tile(tf.expand_dims(clf_tok, 1), [1, max_length, 1])
    clf_tok_x_seq_w_pos = tf.concat((clf_tok_x_seq, flat_pos), -1)
    return tf.concat((X, clf_tok_x_seq_w_pos), 1)

def get_pos_values(seq_len, vocab_size):
    return tf.expand_dims(vocab_size + tf.range(seq_len), 0)

def gpt_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    explain=False,
    **kwargs
):
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
    x_shape = tf.shape(input=X)
    sequence_length = x_shape[1]
    pos_values = get_pos_values(sequence_length, encoder.vocab_size)
    X = tf.stack((X, tf.tile(pos_values, [x_shape[0], 1])), 2)

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

        clf_token = encoder.end_token
        pool_idx = tf.cast(tf.argmax(input=tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), axis=1), tf.int32)

        if explain:
            X = add_explain_tokens(X, sequence_length, pool_idx)

        h = embed(X, embed_weights)
        for layer in range(config.n_layer):
            if (
                (config.n_layer - layer) == config.num_layers_trained
                and config.num_layers_trained != config.n_layer
            ):
                h = tf.stop_gradient(h)
                train_layer = False
            else:
                train_layer = train

            with tf.compat.v1.variable_scope("h%d_" % layer):
                block_fn = functools.partial(
                    block,
                    n_head=config.n_heads,
                    act_fn=config.act_fn,
                    resid_pdrop=config.resid_p_drop,
                    attn_pdrop=config.attn_p_drop,
                    scope="h%d" % layer,
                    train=train_layer,
                    scale=True,
                    explain=explain,
                )
                if config.low_memory_mode and train_layer:
                    block_fn = recompute_grad(block_fn, use_entire_scope=True)
                if layer < config.n_layer - 1:
                    h = block_fn(h)
                else:
                    h_out = block_fn(h)

            # get the attention weights from the last layer
            if layer == config.n_layer - 1:
                with tf.compat.v1.variable_scope("h%d_/h%d/attn" % (layer, layer), reuse=True):
                    q, k, v = multihead_qkv(
                        h, n_state=shape_list(h)[-1], n_head=config.n_heads, train=train
                    )
                    w = attn_weights(q, k, v, scale=True)

        if explain:
            explain_out = h_out[:, initial_shape[1] :]
            explain_out = tf.reshape(
                explain_out, shape=tf.concat((initial_shape, [config.n_embed]), 0)
            )
            h_out = h_out[:, : initial_shape[1]]

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h_out, [-1, config.n_embed])  # [batch * seq_len, embed]
        clf_h = tf.gather(
            clf_h,
            tf.range(shape_list(X)[0], dtype=tf.int32) * sequence_length + pool_idx,
        )
        clf_h = tf.reshape(
            clf_h, shape=tf.concat((initial_shape[:-1], [config.n_embed]), 0)
        )
        seq_feats = tf.reshape(
            h_out, shape=tf.concat((initial_shape, [config.n_embed]), 0)
        )

        lengths = lengths_from_eos_idx(eos_idx=pool_idx, max_length=sequence_length)

        out = {
            "embed_weights": embed_weights,
            "features": clf_h,
            "sequence_features": seq_feats,
            "eos_idx": pool_idx,
            "lengths": lengths,
            "attention_weights": w,  # [n_heads, seq_len, seq_len]
        }
        if explain:
            out["explain_out"] = explain_out
        return out

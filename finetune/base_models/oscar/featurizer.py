import tensorflow as tf
from finetune.base_models.gpt.featurizer import dropout, embed, split_heads, merge_heads
from finetune.util.shapes import shape_list, lengths_from_eos_idx
from finetune.custom_ops import recursive_agg
from finetune.base_models.gpt.featurizer import get_pos_values
from finetune.optimizers.recompute_grads import recompute_grad

import functools


def cast_maybe(t, dtype):
    if t.dtype == dtype:
        return t
    return tf.cast(t, dtype=dtype)


def norm(x, scope, axis=-1, e=None, fp16=False):
    with tf.compat.v1.variable_scope(scope):
        e = e or 1e-5
        n_state = shape_list(x)[-1]
        g = tf.compat.v1.get_variable(
            "g", [n_state], initializer=tf.compat.v1.constant_initializer(1)
        )
        b = tf.compat.v1.get_variable(
            "b", [n_state], initializer=tf.compat.v1.constant_initializer(0)
        )
        if fp16:
            g = tf.cast(g, tf.float16)
            b = tf.cast(b, tf.float16)
        u = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=True)
        s = tf.reduce_mean(input_tensor=tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + e)
        x = x * g + b
        return x


def time_to_batch(value, dilation, pad_with=0):
    with tf.compat.v1.name_scope("time_to_batch"):
        shape = tf.shape(input=value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(
            tensor=value,
            paddings=[[0, 0], [0, pad_elements], [0, 0]],
            constant_values=pad_with,
        )
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(a=reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]]), pad_elements


def batch_to_time(value, dilation):
    with tf.compat.v1.name_scope("batch_to_time"):
        shape = tf.shape(input=value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(a=prepared, perm=[1, 0, 2])
        return tf.reshape(
            transposed, [tf.compat.v1.div(shape[0], dilation), -1, shape[2]]
        )


def cascaded_pool(value, kernel_size, dim=1, pool_len=None, use_fused_kernel=True):
    shape = shape_list(value)
    full_pool_len = pool_len or shape[dim]
    ra = recursive_agg

    aggregated = ra(value, kernel_size, full_pool_len)
    num_pooling_ops = shape_list(aggregated)[2]

    ws = normal_1d_conv_block(
        value, 1, "pool_w", value.dtype == tf.float16, output_dim=shape[-1]
    )
    wt = normal_1d_conv_block(
        value, 1, "pool_t", value.dtype == tf.float16, output_dim=num_pooling_ops
    )
    weights = tf.nn.softmax(wt)
    wt = tf.expand_dims(weights, -1)

    weighted_over_time = tf.reduce_mean(
        input_tensor=aggregated * wt, axis=2
    ) * tf.nn.sigmoid(ws)

    return weighted_over_time


def causal_conv(value, filter_, dilation, name="causal_conv"):
    conv_op = lambda x: tf.nn.conv1d(
        input=x, filters=filter_, stride=1, padding="VALID"
    )
    with tf.compat.v1.name_scope(name):
        if dilation > 1:
            transformed, pad_elements = time_to_batch(value, dilation)
            conv = conv_op(transformed)
            restored = batch_to_time(conv, dilation)
            restored = restored[:, : tf.shape(input=restored)[1] - pad_elements, :]
        else:
            restored = conv_op(value)
        return restored


def cumulative_state_net(
    X,
    name,
    use_fp16,
    pdrop,
    train,
    pool_kernel_size=2,
    nominal_pool_length=512,
    use_fused_kernel=True,
):
    conv_kernel = 4
    pool_kernel_size = pool_kernel_size or conv_kernel

    nx = shape_list(X)[-1]
    with tf.compat.v1.variable_scope(name):
        output = tf.nn.relu(
            normal_1d_conv_block(
                X, conv_kernel, "1-" + str(conv_kernel), use_fp16, output_dim=nx
            )
        )
        output = tf.nn.relu(
            normal_1d_conv_block(
                output, conv_kernel, "2-" + str(conv_kernel), use_fp16, output_dim=nx
            )
        )
        output = normal_1d_conv_block(
            output, conv_kernel, "3-" + str(conv_kernel), use_fp16, output_dim=nx
        )

    output = dropout(output, pdrop, train)
    aggregated = cascaded_pool(
        output,
        kernel_size=pool_kernel_size,
        pool_len=nominal_pool_length,
        use_fused_kernel=use_fused_kernel,
    )

    return normal_1d_conv_block(
        aggregated, 1, "output_reproject", use_fp16, output_dim=nx
    )


def normal_1d_conv_block(
    X, kernel_width, layer_name, use_fp16, dilation=1, output_dim=None, causal=True
):
    # layer_input shape = #batch, seq, embed_dim or batch, channels, seq, embed_dim
    with tf.compat.v1.variable_scope(layer_name):
        # Pad kernel_width (word_wise) - 1 to stop future viewing.
        left_pad = (kernel_width - 1) * dilation

        if causal:
            paddings = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            paddings = [[0, 0], [left_pad // 2, left_pad - (left_pad // 2)], [0, 0]]

        if kernel_width > 1:
            padded_input = tf.pad(tensor=X, paddings=paddings, mode="CONSTANT")
        else:
            padded_input = X

        nx = shape_list(X)[-1]
        if output_dim is None:
            output_dim = nx
        W = tf.compat.v1.get_variable(
            name="W",
            shape=[kernel_width, nx, output_dim],
            initializer=tf.compat.v1.initializers.glorot_normal(),
        )
        b = tf.compat.v1.get_variable(
            name="B",
            shape=[output_dim],
            initializer=tf.compat.v1.initializers.constant(0.0),
        )

        if use_fp16:
            W = tf.cast(W, tf.float16)
            b = tf.cast(b, tf.float16)

        conv = causal_conv(padded_input, W, dilation)
        conv = tf.nn.bias_add(conv, b)
    return conv


def enc_dec_mix(enc, dec, enc_mask, dec_mask, n_head=16):
    # enc = batch, seq, feats
    # dec = batch, seq, feats
    with tf.compat.v1.variable_scope("enc_dec_attn"):
        batch, dec_seq, feats = shape_list(dec)
        enc_seq = shape_list(enc)[1]

        enc_proj = normal_1d_conv_block(
            enc, 1, "enc_proj", use_fp16=enc.dtype == tf.float16, output_dim=feats * 2
        )
        dec_proj = normal_1d_conv_block(
            dec, 1, "dec_proj", use_fp16=enc.dtype == tf.float16, output_dim=feats
        )
        k, v = tf.split(enc_proj, 2, 2)
        q = dec_proj
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        w = tf.matmul(q, k)
        w = w * tf.math.rsqrt(cast_maybe(dec_seq, tf.float32))

        enc_mask = tf.reshape(
            tf.sequence_mask(enc_mask, maxlen=enc_seq, dtype=enc.dtype),
            [batch, 1, 1, enc_seq],
        )
        dec_mask = tf.reshape(
            tf.sequence_mask(dec_mask, maxlen=dec_seq, dtype=enc.dtype),
            [batch, 1, dec_seq, 1],
        )
        m = enc_mask * dec_mask
        w = w * m + -1e9 * (1 - m)
        w = tf.nn.softmax(w)

    return merge_heads(tf.matmul(w, v))


def block(
    X,
    block_name,
    use_fp16,
    pool_idx=None,
    encoder_state=None,
    train=False,
    pdrop=0.1,
    nominal_pool_length=512,
    use_fused_kernel=True,
):
    with tf.compat.v1.variable_scope(block_name):
        h1 = cumulative_state_net(
            X,
            "cumulative_state_net",
            use_fp16,
            pdrop,
            train,
            nominal_pool_length=nominal_pool_length,
            use_fused_kernel=use_fused_kernel,
        )
        if encoder_state is not None:
            mixed = enc_dec_mix(
                encoder_state["sequence_features"],
                h1,
                encoder_state["pool_idx"],
                pool_idx,
            )
            h1 = h1 + mixed
        output = tf.nn.relu(
            dropout(norm(h1 + X, "norm", fp16=use_fp16, e=1e-2), pdrop, train)
        )
        return output


def featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    encoder_state=None,
    context=None,
    context_dim=None,
    **kwargs
):
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
    if len(initial_shape) != 2:
        X = tf.reshape(X, shape=[-1] + initial_shape[-1:])

    x_shape = tf.shape(input=X)
    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        clf_token = encoder.end_token
        pool_idx = tf.cast(
            tf.argmax(input=tf.cast(tf.equal(X, clf_token), tf.float32), axis=1),
            tf.int32,
        )
        if encoder_state is None:
            embed_weights = tf.compat.v1.get_variable(
                "we",
                [encoder.vocab_size + config.max_length, config.n_embed],
                initializer=tf.compat.v1.random_normal_initializer(
                    stddev=config.weight_stddev
                ),
            )
        else:
            embed_weights = encoder_state["embed_weights"]

        if config.oscar_use_fp16:
            embed_weights = tf.cast(embed_weights, tf.float16)

        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        X = tf.reshape(X, [-1, x_shape[1]])
        h = tf.gather(embed_weights, X)
        if config.oscar_use_timing:
            pos_values = get_pos_values(x_shape[1], encoder.vocab_size)
            h += tf.gather(embed_weights, pos_values)

        for layer in range(config.n_layer):
            with tf.compat.v1.variable_scope("h%d_" % layer):
                if (
                    (config.n_layer - layer) == config.num_layers_trained
                    and config.num_layers_trained != config.n_layer
                ):
                    h = tf.stop_gradient(h)

                block_fn_fwd = functools.partial(
                    block,
                    block_name="block%d_" % layer,
                    use_fp16=config.oscar_use_fp16,
                    pool_idx=None,
                    encoder_state=encoder_state,
                    train=train,
                    pdrop=config.resid_p_drop,
                    use_fused_kernel=config.oscar_use_fused_kernel,
                )

                if config.low_memory_mode and train:
                    block_fn_fwd = recompute_grad(block_fn_fwd, use_entire_scope=True)
                h = block_fn_fwd(h)

        h = normal_1d_conv_block(h, 1, "output", config.oscar_use_fp16, dilation=1)

        mask = tf.expand_dims(
            tf.sequence_mask(pool_idx, maxlen=tf.shape(input=h)[1], dtype=h.dtype), -1
        )

        if config.oscar_feat_mode == "clf_tok":
            clf_h = tf.gather_nd(h, tf.stack([tf.range(shape_list(h)[0]), pool_idx], 1))
        elif config.oscar_feat_mode == "mean_tok":
            clf_h = tf.reduce_sum(input_tensor=h * mask, axis=1) / tf.reduce_sum(
                input_tensor=h
            )
        elif config.oscar_feat_mode == "max_tok":
            clf_h = tf.reduce_max(input_tensor=h - (1e5 * (1.0 - mask)), axis=1)
        else:
            raise ValueError(
                "config.feat_mode should be one of clf_tok, mean_tok or max_tok"
            )

        if len(initial_shape) != 2:
            seq_feats = tf.reshape(h, shape=initial_shape + [config.n_embed])
        else:
            seq_feats = h

        return {
            "embed_weights": embed_weights,
            "features": cast_maybe(clf_h, tf.float32),
            "sequence_features": seq_feats,
            "eos_idx": pool_idx,
            "encoded_input": X[:, : tf.reduce_min(input_tensor=pool_idx)],
            "lengths": lengths_from_eos_idx(
                eos_idx=pool_idx, max_length=shape_list(X)[1]
            ),
        }

import tensorflow as tf

from finetune.base_models.common import (
    MLP,
    Conv1D,
    embed,
    get_pos_values,
    merge_heads,
    split_heads,
)
from finetune.nn.nn_utils import ExtraScope, Norm, maybe_recompute
from finetune.util.shapes import shape_list


def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.linalg.band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w * b + -1e9 * (1 - b)
    return w


def mask_pad(w, lengths):
    batch = shape_list(lengths)[0]
    maxlen = tf.cast(tf.reduce_max(input_tensor=lengths), tf.int32)
    seq_mask = tf.reshape(
        tf.sequence_mask(lengths, maxlen=maxlen), [batch, 1, 1, maxlen]
    )
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


class Attn(tf.keras.layers.Layer):
    def __init__(self, *, num_heads, attn_pdrop, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(attn_pdrop)

    def call(self, x):
        c = self.c_attn(x)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads, k=True)
        v = split_heads(v, self.num_heads)
        w = attn_weights(q, k, v)
        w = self.dropout(w)
        a = tf.matmul(w, v)
        a = merge_heads(a)
        a = self.c_proj(a)
        a = self.dropout(a)
        return a

    def build(self, input_shape):
        # Unfortunate to put these in build but with the *3 we are dependent on the input shape
        self.c_attn = Conv1D(
            num_filters=input_shape[-1] * 3, kernel_size=1, name="c_attn"
        )
        self.c_proj = Conv1D(num_filters=input_shape[-1], kernel_size=1, name="c_proj")
        super().build(input_shape)


class Block(tf.keras.layers.Layer):
    def __init__(self, *, n_head, act_fn, resid_pdrop, attn_pdrop, **kwargs):
        super().__init__(**kwargs)
        self.ln_1 = Norm(name="ln_1")
        self.attn = Attn(num_heads=n_head, attn_pdrop=attn_pdrop, name="attn")
        self.ln_2 = Norm(name="ln_2")
        self.resid_pdrop = resid_pdrop
        self.act_fn = act_fn

    def build(self, input_shape):
        self.mlp = MLP(
            n_state=input_shape[-1] * 4,
            resid_pdrop=self.resid_pdrop,
            act_fn=self.act_fn,
            name="mlp",
        )
        super().build(input_shape)

    def call(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        return self.ln_2(n + m)


class GPTFeaturizer(tf.keras.layers.Layer):
    def __init__(self, encoder, config, **kwargs):
        super().__init__(**kwargs)

        self.blocks = [
            ExtraScope(
                layer=Block(
                    n_head=config.n_heads,
                    act_fn=config.act_fn,
                    resid_pdrop=config.resid_p_drop,
                    attn_pdrop=config.attn_p_drop,
                    name=f"h{i}",
                ),
                name=f"h{i}_",
            )
            for i in range(config.n_layer)
        ]
        self.embed_dropout = tf.keras.layers.Dropout(config.embed_p_drop)
        self.vocab_size = encoder.vocab_size
        self.max_length = config.max_length
        self.n_embed = config.n_embed
        self.clf_token = encoder.end_token
        self.weight_stddev = config.weight_stddev
        self.do_recompute = config.low_memory_mode

    def build(self, input_shape):
        self.embed_weights = self.add_weight(
            shape=[self.vocab_size + self.max_length, self.n_embed],
            initializer=tf.keras.initializers.RandomNormal(stddev=self.weight_stddev),
            trainable=True,
            name="we",
        )
        super().build(input_shape)

    def call(self, tokens, context, sequence_lengths, training=True):
        tokens_shape = tf.shape(tokens)
        batch_size, seq_dim = tokens_shape[0], tokens_shape[1]
        pos_values = get_pos_values(seq_dim, self.vocab_size)
        pool_idx = tf.cast(
            tf.argmax(
                input=tf.cast(tf.equal(tokens, self.clf_token), tf.float32), axis=1
            ),
            tf.int32,
        )
        tokens_with_pos = tf.stack((tokens, tf.tile(pos_values, [batch_size, 1])), 2)

        embed_weights = self.embed_dropout(self.embed_weights)
        h = embed(tokens_with_pos, embed_weights)
        for block in self.blocks:
            h = maybe_recompute(
                block, do_recompute=self.do_recompute, training=training
            )(h)
        clf_h = tf.reshape(h, [-1, self.n_embed])  # [batch * seq_len, embed]
        clf_h = tf.gather(
            clf_h,
            tf.range(batch_size, dtype=tf.int32) * sequence_lengths + pool_idx,
        )
        return {
            "features": clf_h,
            "sequence_features": h,
        }

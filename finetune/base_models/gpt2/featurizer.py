import tensorflow as tf

from finetune.base_models.common import (
    MLP,
    Conv1D,
    embed,
    get_pos_values,
    merge_heads,
    split_heads,
)
from finetune.nn.nn_utils import Norm, maybe_recompute
from finetune.util.shapes import shape_list


def softmax(x, axis=-1):
    x = x - tf.reduce_max(input_tensor=x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(input_tensor=ex, axis=axis, keepdims=True)


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


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
    # Rather than refactor this going to go for a lazy style keras call
    w = tf.keras.layers.Dropout(attn_p_drop)(w, training=train)
    a = tf.matmul(w, v)
    return a


class Attn(tf.keras.layers.Layer):
    def __init__(self, *, attn_p_drop, resid_p_drop, **kwargs):
        super().__init__(**kwargs)
        self.attn_p_drop = attn_p_drop
        self.resid_p_drop = resid_p_drop
        self.resid_drop = tf.keras.layers.Dropout(resid_p_drop)

    def build(self, input_shape):
        assert input_shape.ndims == 3
        self.c_attn = Conv1D(
            num_filters=input_shape[-1] * 3, kernel_size=1, name="c_attn"
        )
        self.c_proj = Conv1D(num_filters=input_shape[-1], kernel_size=1, name="c_proj")

    def call(self, x, past=None, training=False):
        c = self.c_attn(x)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v, self.attn_p_drop, train=training)
        a = merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_drop(a)
        return a


class Block(tf.keras.layers.Layer):
    def __init__(self, attn_p_drop, resid_p_drop, **kwargs):
        super().__init__(**kwargs)
        self.attn_p_drop = attn_p_drop
        self.resid_p_drop = resid_p_drop
        self.norm = Norm(name="ln_1")
        self.attn = Attn(
            attn_p_drop=attn_p_drop, resid_p_drop=resid_p_drop, name="attn"
        )

    def build(self, input_shape):
        self.mlp = MLP(
            n_state=input_shape[-1] * 4,
            resid_pdrop=self.resid_p_drop,
            act_fn="gelu",
            name="mlp",
        )

    def call(self, x, past=None):
        a = self.attn(self.norm(x), past=past)
        x = x + a
        m = self.mlp(self.norm(x, "ln_2"))
        x = x + m
        return x


class GPT2Featurizer(tf.keras.layers.Layer):
    def __init__(self, encoder, config, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            Block(
                attn_p_drop=config.attn_p_drop,
                resid_p_drop=config.attn_p_drop,
                name="h%d" % i,
            )
            for i in range(config.n_layer)
        ]
        self.embed_dropout = tf.keras.layers.Dropout(config.embed_p_drop)
        self.vocab_size = encoder.vocab_size
        self.max_length = config.max_length
        self.n_embed = config.n_embed
        self.clf_token = encoder["_classify_"]
        self.weight_stddev = config.weight_stddev
        self.do_recompute = config.low_memory_mode
        self.ln_f = Norm(name="ln_f")

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
        h = self.ln_f(block)
        clf_h = tf.reshape(h, [-1, self.n_embed])  # [batch * seq_len, embed]
        clf_h = tf.gather(
            clf_h,
            tf.range(batch_size, dtype=tf.int32) * sequence_lengths + pool_idx,
        )
        return {
            "features": clf_h,
            "sequence_features": h,
        }

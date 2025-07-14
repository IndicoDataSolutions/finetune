import numpy as np
import tensorflow as tf

from finetune.util.shapes import shape_list
from finetune.nn.activations import act_fns
from finetune.nn.nn_utils import Norm, ExtraScope, maybe_recompute


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

class Conv1D(tf.keras.layers.Layer):
    def __init__(self, *,  num_filters: int | None, kernel_size: int, pad="VALID", **kwargs):
        super().__init__(**kwargs)
        self.pad = pad
        self.kernel_size = kernel_size
        self.num_filters = num_filters

    def call(self, x):
        nx = shape_list(x)[-1]
        if self.kernel_size == 1:  # faster 1x1 conv
            c = tf.reshape(
                tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(self.w, [-1, self.num_filters])) + self.b,
                shape_list(x)[:-1] + [self.num_filters],
            )
        else:  # was used to train LM
            c = tf.nn.conv1d(input=x, filters=self.w, stride=1, padding=self.pad) + self.b
        return c

    def build(self, input_shape):
        if self.num_filters is None:
            self.num_filters = input_shape[-1]
        self.w = self.add_weight(
            shape=(self.kernel_size, input_shape[-1], self.num_filters),
            initializer="random_normal",
            trainable=True,
            name="w"
        )
        self.b = self.add_weight(
            shape=(self.num_filters,),
            initializer="zeros",
            trainable=True,
            name="b"
        )
        super().build(input_shape)


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
        self.c_attn = Conv1D(num_filters=input_shape[-1] * 3, kernel_size=1, name="c_attn")
        self.c_proj = Conv1D(num_filters=input_shape[-1], kernel_size=1, name="c_proj")
        super().build(input_shape)




class MLP(tf.keras.layers.Layer):
    def __init__(self, *, n_state, resid_pdrop, act_fn, **kwargs):
        super().__init__(**kwargs)
        self.act_fn = act_fn
        self.c_fc = Conv1D(num_filters=n_state, kernel_size=1, name="c_fc")
        self.dropout = tf.keras.layers.Dropout(resid_pdrop)

    def build(self, input_shape):
        self.c_proj = Conv1D(num_filters=input_shape[-1], kernel_size=1, name="c_proj")
        super().build(input_shape)

    def call(self, x):
        h = self.c_fc(x)
        h = act_fns[self.act_fn](h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h

class Block(tf.keras.layers.Layer):
    def __init__(self, *, n_head, act_fn, resid_pdrop, attn_pdrop, **kwargs):
        super().__init__(**kwargs)
        self.ln_1 = Norm(name="ln_1")
        self.attn = Attn(num_heads=n_head, attn_pdrop=attn_pdrop, name="attn")
        self.ln_2 = Norm(name="ln_2")
        self.resid_pdrop = resid_pdrop
        self.act_fn = act_fn

    def build(self, input_shape):
        self.mlp = MLP(n_state=input_shape[-1] * 4, resid_pdrop=self.resid_pdrop, act_fn=self.act_fn, name="mlp")
        super().build(input_shape)


    def call(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        return self.ln_2(n + m)


def embed(X, we):
    return tf.reduce_sum(input_tensor=tf.gather(we, X), axis=2)


def get_pos_values(seq_len, vocab_size):
    return tf.expand_dims(vocab_size + tf.range(seq_len), 0)

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
                    name=f"h{i}"
                ),
                name=f"h{i}_"
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
            name="we"
        )
        super().build(input_shape)
        
    def call(self, tokens, context, sequence_lengths, training=True):
        tokens_shape = tf.shape(tokens)
        batch_size, seq_dim = tokens_shape[0], tokens_shape[1]
        pos_values = get_pos_values(seq_dim, self.vocab_size)
        pool_idx = tf.cast(tf.argmax(input=tf.cast(tf.equal(tokens, self.clf_token), tf.float32), axis=1), tf.int32)
        tokens_with_pos = tf.stack((tokens, tf.tile(pos_values, [batch_size, 1])), 2)

        embed_weights = self.embed_dropout(self.embed_weights)
        h = embed(tokens_with_pos, embed_weights)
        for block in self.blocks:
            h = maybe_recompute(block, do_recompute=self.do_recompute, training=training)(h)
        clf_h = tf.reshape(h, [-1, self.n_embed])  # [batch * seq_len, embed]
        clf_h = tf.gather(
            clf_h,
            tf.range(batch_size, dtype=tf.int32) * sequence_lengths + pool_idx,
        )
        return {
            "features": clf_h,
            "sequence_features": h,
        }

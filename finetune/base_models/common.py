import numpy as np
import math
import tensorflow as tf

from finetune.nn.activations import gpt_gelu
from finetune.util.shapes import shape_list


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


def gelu(x):
    return (
        0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    )


class Conv1D(tf.keras.layers.Layer):
    def __init__(
        self, *, num_filters: int | None, kernel_size: int, pad="VALID", **kwargs
    ):
        super().__init__(**kwargs)
        self.pad = pad
        self.kernel_size = kernel_size
        self.num_filters = num_filters

    def call(self, x):
        nx = shape_list(x)[-1]
        if self.kernel_size == 1:  # faster 1x1 conv
            c = tf.reshape(
                tf.matmul(
                    tf.reshape(x, [-1, nx]), tf.reshape(self.w, [-1, self.num_filters])
                )
                + self.b,
                shape_list(x)[:-1] + [self.num_filters],
            )
        else:  # was used to train LM
            c = (
                tf.nn.conv1d(input=x, filters=self.w, stride=1, padding=self.pad)
                + self.b
            )
        return c

    def build(self, input_shape):
        if self.num_filters is None:
            self.num_filters = input_shape[-1]
        self.w = self.add_weight(
            shape=(self.kernel_size, input_shape[-1], self.num_filters),
            initializer="random_normal",
            trainable=True,
            name="w",
        )
        self.b = self.add_weight(
            shape=(self.num_filters,), initializer="zeros", trainable=True, name="b"
        )
        super().build(input_shape)


class MLP(tf.keras.layers.Layer):
    def __init__(self, *, n_state, resid_pdrop, **kwargs):
        super().__init__(**kwargs)
        self.c_fc = Conv1D(num_filters=n_state, kernel_size=1, name="c_fc")
        self.dropout = tf.keras.layers.Dropout(resid_pdrop)

    def build(self, input_shape):
        self.c_proj = Conv1D(num_filters=input_shape[-1], kernel_size=1, name="c_proj")
        super().build(input_shape)

    def call(self, x):
        h = self.c_fc(x)
        h = gpt_gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h


def embed(X, we):
    return tf.reduce_sum(input_tensor=tf.gather(we, X), axis=2)


def get_pos_values(seq_len, vocab_size):
    return tf.expand_dims(vocab_size + tf.range(seq_len), 0)

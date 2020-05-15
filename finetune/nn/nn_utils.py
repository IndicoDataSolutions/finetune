import tensorflow as tf
from finetune.util.shapes import shape_list


def norm(x, scope, axis=[-1], e=1e-5):
    with tf.compat.v1.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.compat.v1.get_variable('g', [n_state], initializer=tf.compat.v1.constant_initializer(1))
        b = tf.compat.v1.get_variable('b', [n_state], initializer=tf.compat.v1.constant_initializer(0))
        u = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=True)
        s = tf.reduce_mean(input_tensor=tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + e)
        x = x * g + b
        return x


def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1 - (1 - pdrop))
    return x

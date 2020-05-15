import os
import math

import tensorflow as tf

from finetune.custom_ops.indico_tf_ops.python import indico_ops

def _causalmax(x, dim, pool_len, padding_type="left"):
    kernel_size = [1, 1, 1]
    pool_len = pool_len
    kernel_size[dim] = pool_len
    padding = [[0, 0], [0, 0], [0, 0]]
    if padding_type == "left":
        padding[dim] = [pool_len - 1, 0]
    elif padding_type == "right":
        padding[dim] = [0, pool_len - 1]
    elif padding_type == "even":
        leng = pool_len - 1
        padding[dim] = [leng // 2, leng - leng // 2]
    else:
        raise ValueError("Padding type {} not supported".format(padding_type))
    padded_x = tf.pad(x, padding, "CONSTANT", constant_values=-1e4)

    kernel_size.insert(2, 1)

    return tf.squeeze(
        tf.nn.max_pool(
            value=tf.expand_dims(padded_x, 2),
            ksize=kernel_size,
            strides=[1, 1, 1, 1],
            padding="VALID"
        ),
        axis=2
    )


def _time_to_batch(value, dilation, pad_with=0):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]], constant_values=pad_with)
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]]), pad_elements


def batch_to_time(value, dilation):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def _dilated_causal_max_pool(value, kernel_size, dilation, padding="left"):
    pool_op = lambda x, padding: _causalmax(x, dim=1, pool_len=kernel_size, padding_type=padding)
    with tf.name_scope("causal_pool"):
        if dilation > 1:
            transformed, pad_elements = _time_to_batch(value, dilation, pad_with=-1e4)
            pooled = pool_op(transformed, padding)
            restored = batch_to_time(pooled, dilation)
            restored = restored[:, :tf.shape(restored)[1] - pad_elements, :]
        else:
            restored = pool_op(value, padding)

        restored.set_shape(value.get_shape())
        return restored


def recursive_agg(value, kernel_size, pool_len, causal=True):
    if indico_ops.BUILT:
        return indico_ops.recursive_agg_op(value, kernel_size, pool_len)

    full_pool_len = pool_len
    num_pooling_ops = int(math.ceil(math.log(full_pool_len, kernel_size)))
    
    intermediate_vals = []
    for i in range(num_pooling_ops):
        value = _dilated_causal_max_pool(value, kernel_size=kernel_size, dilation=kernel_size ** i,
                                         padding="left" if causal else "even")
        intermediate_vals.append(value)
    
    return tf.stack(intermediate_vals, 2)

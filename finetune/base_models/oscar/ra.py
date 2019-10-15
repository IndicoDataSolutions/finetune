import os
import math
import warnings

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import NotFoundError


def _causalmax(x, dim, pool_len):
    kernel_size = [1, 1, 1]
    pool_len = pool_len
    kernel_size[dim] = pool_len
    padding = [[0, 0], [0, 0], [0, 0]]
    padding[dim] = [pool_len - 1, 0]
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


def _dilated_causal_max_pool(value, kernel_size, dilation):
    pool_op = lambda x: _causalmax(x, dim=1, pool_len=kernel_size)
    with tf.name_scope("causal_pool"):
        if dilation > 1:
            transformed, pad_elements = _time_to_batch(value, dilation, pad_with=-1e4)
            pooled = pool_op(transformed)
            restored = batch_to_time(pooled, dilation)
            restored = restored[:, :tf.shape(restored)[1] - pad_elements, :]
        else:
            restored = pool_op(value)

        restored.set_shape(value.get_shape())
        return restored


def recursive_agg_tf(value, kernel_size, pool_len):
    full_pool_len = pool_len
    num_pooling_ops = int(math.ceil(math.log(full_pool_len, kernel_size)))
    
    intermediate_vals = []
    for i in range(num_pooling_ops):
        value = _dilated_causal_max_pool(value, kernel_size=kernel_size, dilation=kernel_size ** i)
        intermediate_vals.append(value)

    return tf.stack(intermediate_vals, 2)


try:
    # TODO: figure out how to ship this in a way that will build with finetune.
    ra_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'lib_ra.so'))

    @ops.RegisterGradient("Dense")
    def _grad_cc(op, grad0, grad1):
        return ra_module.dense_grad(grad0, op.inputs[0], op.outputs[1])

    def recursive_agg(inp, kernel_length, pool_len):
        return ra_module.dense(inp, kernel_length, pool_len, int(math.ceil(math.log(pool_len, kernel_length))))[0]

except NotFoundError:
    warnings.warn("Failed to load Oscar's optimised kernels. Falling back to tensorflow version.")
    recursive_agg = recursive_agg_tf

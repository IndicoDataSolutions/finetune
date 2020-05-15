import os
import math

import numpy as np
import tensorflow as tf

from finetune.util.shapes import shape_list
from finetune.custom_ops.indico_tf_ops.python import indico_ops

def linear(inp, output_dim, layer_name):
    with tf.variable_scope(layer_name):
        nx = shape_list(inp)[-1]
        if output_dim is None:
            output_dim = nx
        W = tf.get_variable(
            name="W", shape=[nx, output_dim], initializer=tf.initializers.glorot_normal()
        )
        if inp.dtype == tf.float16:
            W = tf.cast(W, tf.float16)
                
        return tf.reshape(
            tf.matmul(
                tf.reshape(inp, [-1, nx]),
                tf.reshape(W, [nx, output_dim])
            ),
            shape_list(inp)[:-1] + [output_dim]
        )
            
def unfold(inp, kernel_width, padding="causal"):
    if padding.lower() == "causal":
        padding_l = kernel_width - 1
    elif padding.lower() == "same":
        padding_l = kernel_width // 2
                        
    slices = []
    paddings = [[0, 0], [padding_l, kernel_width - padding_l - 1], [0, 0]]
    padded_input = tf.pad(inp, paddings, "CONSTANT")
    leng = tf.shape(padded_input)[1]
    for i in range(kernel_width):
        slices.append(padded_input[:, i: leng - kernel_size + i + 1])
    return tf.stack(slices, 2)
    
def dynamic_conv_cpu(inp, weights, padding="causal"):
    """
    inp : Batch Time Channels
    weights: Batch, Time, FilterWidth, Heads
    padding: causal or same
    """
    batch, seq, n_channels = shape_list(inp)
    _, _, kernel_width, n_heads = shape_list(weights)
    assert n_channels % n_heads == 0
    h = n_channels // n_heads
    
    unfolded = unfold(inp, kernel_width, padding="causal")
    unfolded = tf.reshape(unfolded, [batch, seq, kernel_size, n_heads, h])
    weights_expanded = tf.expand_dims(weights, 4)
    return tf.reshape(tf.reduce_sum(weights_expanded * unfolded, 2), shape_list(inp))

def dynamic_conv_on_ra_out(ra_out, weights):
    """
    ra_out: batch, length, ra_size, channels
    weights: atch, Time, FilterWidth, Heads
    """
    batch, seq, ra_depth, n_channels = shape_list(ra_out)
    _, _, kernel_width, n_heads = shape_list(weights)
    assert n_channels % n_heads == 0
    assert ra_depth == kernel_width
    h = n_channels // n_heads
    unfolded = tf.reshape(ra_out, [batch, seq, ra_depth, n_heads, h])
    weights_expanded = tf.expand_dims(weights, 4)
    return tf.reshape(tf.reduce_sum(weights_expanded * unfolded, 2), [batch, seq, n_channels])

def dynamic_convolution(inp, inp2=None, n_heads=8, kernel_size=4, padding="causal"):
    batch, seq, n_channels = shape_list(inp)
    kernel_size_inc_ra = kernel_size
    if inp2 is not None:
        kernel_size_inc_ra += shape_list(inp2)[2]
        
    weights_linear = tf.reshape(
        linear(inp, n_heads * kernel_size_inc_ra, "kernel_machine",),
        [batch, seq, kernel_size_inc_ra, n_heads], # batch time heads, kernel_size, 1
    )
    weights_linear = tf.nn.softmax(weights_linear, 2)

    if inp2 is not None:
        weights_ra = weights_linear[:, :, kernel_size:]
        weights_linear = weights_linear[:, :, :kernel_size]
        ra_dynamic = dynamic_conv_on_ra_out(inp2, weights_ra)
    else:
        ra_dynamic = 0.0

    if indico_ops.BUILT and tf.test.is_gpu_available():
        # There is no CPU version of this op we need to check this.
        dynamic_conv_fn = indico_ops.dynamic_convolution_op 
    else:
        dynamic_conv_fn = dynamic_conv_cpu

    return dynamic_conv_fn(inp, weights_linear, padding=padding) + ra_dynamic
    
        
        
        

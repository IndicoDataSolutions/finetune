import math

import tensorflow as tf	
from finetune.util.shapes import shape_list


def add_timing_signal_from_position(x, position, timescales):
    """
    Args:
      x: a Tensor with shape [batch, len, channels]
      position: [batch, len, nd]
      min_timescale: a float
      max_timescale: a float
    Returns:
      a Tensor the same shape as x.
    """
    channels = shape_list(x)[2]
    num_dims = shape_list(position)[2]

    num_timescales = channels // (num_dims * 2)

    for dim, timescale in zip(range(num_dims), timescales):
        min_timescale, max_timescale = timescale
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / (tf.cast(num_timescales, dtype=tf.float32) - 1)
        )
        inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), dtype=tf.float32) * log_timescale_increment)
        position_x = tf.expand_dims(tf.cast(position[:, :, dim], dtype=tf.float32), 2)  # batch, len, 1 # where 1 will be the chanels dim
        scaled_time = position_x * tf.expand_dims(tf.expand_dims(inv_timescales, 0), 0)  # batch , len, num_timescales
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)  # batch channels//num_dims
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(tensor=signal, paddings=[[0, 0], [0, 0], [prepad, postpad]])
        x = x + signal
    return x

def embed_position(context, context_channels, batch, seq):
    with tf.compat.v1.variable_scope("context_embedding"):
        context_dim = shape_list(context)[-1]
        if context_channels is None:
            raise ValueError("context_channels is not set but you are trying to embed context")
        x = tf.zeros(shape=(batch, seq, context_channels))

        def get_pos_embed():
            return add_timing_signal_from_position(
                x,
                context,
                timescales = [
                    [
                        (math.pi / 2) * (1 / 2500),
                        (25 * math.pi) * (1 / 2500)
                    ]
                ] * context_dim
            ) / (float(context_channels) / 32)
        def identity():
            return x
        pos_embed = tf.cond(
            tf.equal(seq, 0), true_fn=identity, false_fn=get_pos_embed
        )
    return pos_embed


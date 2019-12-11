import math

import numpy as np
from scipy import interpolate
import tensorflow as tf

from finetune.util.shapes import shape_list


def interpolate_pos_embed(positional_embed, new_len):
    xx = np.linspace(0, 512, new_len)
    newKernel = interpolate.RectBivariateSpline(
        np.arange(positional_embed.shape[0]),
        np.arange(positional_embed.shape[1]),
        positional_embed,
    )
    return newKernel(xx, np.arange(positional_embed.shape[1]))


def process_pos_embed(positional_embed, max_length, interpolate):
    if interpolate and max_length != len(positional_embed):
        positional_embed = interpolate_pos_embed(positional_embed, max_length)

    elif max_length > len(positional_embed):
        raise ValueError(
            "Max Length cannot be greater than {} if interpolate_pos_embed is turned off".format(
                len(positional_embed)
            )
        )
    else:
        positional_embed = positional_embed[:max_length]
    return positional_embed


def embedding_preprocessor(input_pipeline, config):
    def process_embeddings(name, value):
        if "/we:0" in name:
            vocab_size = input_pipeline.text_encoder.vocab_size
            word_embeddings = value[: vocab_size]
            positional_embed = value[vocab_size:]

            positional_embed = process_pos_embed(
                positional_embed, config.max_length, config.interpolate_pos_embed
            )
            value = np.concatenate(
                (word_embeddings, positional_embed), axis=0
            )

        elif "position_embeddings" in name:
            length = config.max_length
            if "roberta" in config.base_model.__name__.lower():
                length += 2
            value = process_pos_embed(value, length, config.interpolate_pos_embed)

        return value

    return process_embeddings


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
    
    for i, something in enumerate(zip(range(num_dims), timescales)):
        dim, timescale = something
        print(i)
        print('dim', dim)
        min_timescale, max_timescale = timescale
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * log_timescale_increment)
        position_x = tf.expand_dims(tf.to_float(position[:, :, dim]), 2)  # batch, len, 1 # where 1 will be the chanels dim
        scaled_time = position_x * tf.expand_dims(tf.expand_dims(inv_timescales, 0), 0)  # batch , len, num_timescales
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)  # batch channels//num_dims
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [0, 0], [prepad, postpad]])
        x = x + signal
    # x = tf.Print(x, [x], summarize=10000)
    return x


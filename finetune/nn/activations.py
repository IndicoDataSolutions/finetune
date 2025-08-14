"""
Activation functions
"""
import math

import numpy as np
import tensorflow as tf


def swish(x):
    return x * tf.nn.sigmoid(x)


def bert_gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.

    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def gpt_gelu(x):
    return (
        0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    )


def _modernbert_gelu(x):
    # This matches the old tensorflow exact implementation for fp16
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


def modernbert_gelu(x):
    if x.dtype != tf.float32:
        # Previously approximate form was used for non fp32
        return tf.cast(tf.nn.gelu(tf.cast(x, tf.float32), approximate=True), x.dtype)
    return _modernbert_gelu(x)


hf_gelu = tf.keras.activations.gelu

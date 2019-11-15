import math
import tensorflow as tf
from itertools import zip_longest

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context


def warmup_cosine(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))

def warmup_constant(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1

def warmup_linear(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)

def exp_decay_oscar(x, warmup=0.001):
    s = tf.cast(x <= warmup, tf.float32)
    return s * (x / warmup) + (1-s) * (1 / (1.005 ** (1000 * (x - warmup) / (1 - warmup))))

schedules = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
    'exp_decay_oscar': exp_decay_oscar,
    'none': lambda x, *args, **kwargs: x
}


import math

import tensorflow as tf


def warmup_cosine(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return s * (x / warmup) + (1 - s) * (0.5 * (1 + tf.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return s * (x / warmup) + (1 - s) * 1


def warmup_linear(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return (s * (x / warmup) + (1 - s)) * (1 - x)


def exp_decay_oscar(x, warmup=0.001):
    s = tf.cast(x <= warmup, tf.float32)
    return s * (x / warmup) + (1 - s) * (
        1 / (1.005 ** (1000 * (x - warmup) / (1 - warmup)))
    )


schedules = {
    "warmup_cosine": warmup_cosine,
    "warmup_constant": warmup_constant,
    "warmup_linear": warmup_linear,
    "exp_decay_oscar": exp_decay_oscar,
    "none": lambda x, *args, **kwargs: x,
}


class FinetuneKerasLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, schedule: str, base_lr: float, total_steps: int, warmup: float = None
    ):
        self.schedule = schedules[schedule]
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup = warmup
        self.kwargs = {"warmup": warmup} if warmup is not None else {}

    def __call__(self, step):
        lr = self.base_lr * self.schedule(step / self.total_steps, **self.kwargs)
        return tf.maximum(lr, 0.0)

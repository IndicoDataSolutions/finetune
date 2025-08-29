import tensorflow as tf


class Norm(tf.keras.layers.Layer):
    def __init__(self, axis=[-1], e=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.e = e

    def build(self, input_shape):
        self.g = self.add_weight(
            name="g",
            shape=[input_shape[-1]],
            initializer=tf.compat.v1.constant_initializer(1),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=[input_shape[-1]],
            initializer=tf.compat.v1.constant_initializer(0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        u = tf.reduce_mean(input_tensor=x, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(input_tensor=tf.square(x - u), axis=self.axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + self.e)
        x = x * self.g + self.b
        return x


class ExtraScope(tf.keras.layers.Layer):
    def __init__(self, layer, name, **kwargs):
        super().__init__(**kwargs, name=name)
        self.layer = layer

    def build(self, input_shape):
        # Just to silence the internal warnings, because nothing is included in this layer.
        super().build(input_shape)

    def call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        # Pass this through so we can wrap target model without running into issues.
        return self.layer.compute_loss(*args, **kwargs)


def saver_ignore_scope(cls: tf.keras.layers.Layer):
    """
    Helper decorator to mark layers that should not be considered scopes.
    Just allows us to structure layers more nicely when we're porting models that had a flat structure.
    """
    # Might want to do somethign different here long term but for now this is good enough.
    cls._saver_ignore_scope = True
    return cls


def maybe_recompute(fn, do_recompute, training):
    if do_recompute and training:
        return tf.recompute_grad(fn)
    return fn

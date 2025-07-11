import tensorflow as tf

class Norm(tf.keras.layers.Layer):
    def __init__(self, feature_size: int, axis=[-1], e=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.e = e
        self.g = self.add_weight(
            name="g",
            shape=[feature_size],
            initializer=tf.compat.v1.constant_initializer(1),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=[feature_size],
            initializer=tf.compat.v1.constant_initializer(0),
            trainable=True,
        )

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

    def call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

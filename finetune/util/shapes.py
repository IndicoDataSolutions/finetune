import tensorflow as tf


# Tensorflow
def merge_leading_dims(X, target_rank):
    shape = [-1] + X.get_shape().as_list()[1 - target_rank:]
    return tf.reshape(X, shape)


def concat_or_stack(tensors, axis=0):
    try:
        return tf.concat(tensors, axis=axis)
    except ValueError:
        # tensors are scalars
        return tf.stack(tensors, axis=axis)


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


# Python
def flatten(outer):
    return [el for inner in outer for el in inner]


def list_transpose(l):
    return [list(i) for i in zip(*l)]

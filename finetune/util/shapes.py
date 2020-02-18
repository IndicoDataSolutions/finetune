import tensorflow as tf


def shape_list(x):
    """
    Deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def concat_or_stack(tensors, axis=0):
    try:
        return tf.concat(tensors, axis=axis)
    except ValueError:
        # tensors are scalars
        return tf.stack(tensors, axis=axis)


def merge_leading_dims(X, target_rank):
    original_shape = X.get_shape().as_list()
    original_rank = len(original_shape)
    if target_rank > original_rank:
        raise ValueError("Original rank is less than the target rank")
    
    if original_rank == target_rank:
        return X
    output_shape = [-1] + original_shape[original_rank - target_rank + 1:]
    return tf.reshape(X, output_shape)


def lengths_from_eos_idx(eos_idx, max_length):
    return tf.where(
        tf.equal(eos_idx, 0),
        tf.ones_like(eos_idx) * tf.cast(max_length, dtype=eos_idx.dtype),
        eos_idx + 1,
    )

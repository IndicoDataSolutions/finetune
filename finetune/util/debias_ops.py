import numpy as np
import tensorflow as tf


def get_shape_tuple(x, axis=None):
  """Shape of a tensor as a tuple composed of ints or scalar int tensors"""
  s = x.shape.as_list()
  if axis is None:
    tf_shape = tf.shape(x)
    for i in range(len(s)):
      if s[i] is None:
        s[i] = tf_shape[i]
    return s
  if s[axis] is None:
    return tf.shape(x)[axis]
  else:
    return s[axis]


def flatten(x):
  """Flatten x while trying to preserve shape information"""
  sh = x.shape.as_list()
  if any(x is None for x in sh):
    return tf.reshape(x, [-1])
  else:
    return tf.reshape(x, [np.prod(sh)])


def affine(x, output_size: int, weight_name: str, bias_name=None, weight_init=None):
  """Affine transformation of `x` using the specified variable names"""
  dim = x.shape.as_list()[-1]
  w = tf.get_variable(
      weight_name, (dim, output_size), tf.float32, initializer=weight_init)
  out = tf.tensordot(x, w, [[len(x.shape) - 1], [0]])

  if bias_name:
    b = tf.get_variable(
      bias_name, (output_size,), tf.float32, initializer=tf.zeros_initializer())
    for _ in range(len(out.shape) - 1):
      b = tf.expand_dims(b, 0)
    out += b
  return out


def last_dim_weighted_sum(x, weight_name, weight_init=None, keepdims=False):
  """Weighted sum of the last dim of a tensor using the given weight name"""
  dim = x.shape.as_list()[-1]
  w = tf.get_variable(weight_name, dim, initializer=weight_init)
  out = tf.tensordot(x, w, [[len(x.shape) - 1], [0]])
  if keepdims:
    return tf.expand_dims(out, len(out.shape))
  else:
    return out


def entropy(logits, mask=None):
  """Compute the entropy of the probabilities implied by `logit`"""
  logits = tf.nn.log_softmax(mask_logits(logits, mask))
  prob = tf.exp(logits)
  if mask is not None:
    f_mask = tf.sequence_mask(mask, tf.shape(logits)[1], tf.float32)
    return -tf.reduce_mean(tf.reduce_sum(prob * (logits * f_mask), -1))
  else:
    return -tf.reduce_mean(tf.reduce_sum(prob * logits, -1))


def max_pool(x, mask):
  """Max pool along dimension 1 followed by RELu"""
  if mask is not None:
    mask = tf.sequence_mask(mask, get_shape_tuple(x, 1), tf.float32)
    mask = tf.expand_dims(mask, 2)
    x *= mask
  return tf.maximum(tf.reduce_max(x, -2), 0)


def bucket_by_quantiles(len_fn, batch_size, n_buckets, hist_bounds):
  n_hist_binds = len(hist_bounds)

  if n_hist_binds < n_buckets:
    raise ValueError("Requested %d buckets, but only have %d histogram bins" %
                     (n_buckets, n_hist_binds))
  if any(hist_bounds[i] >= hist_bounds[i+1] for i in range(n_hist_binds-1)):
    raise ValueError("Bins must be descending")

  # The hisogram: A count of the number of elements whose length was
  # greater than a fixed set values (so `hist_counts[i]` is the number of elements
  # with size > hist_bounds[i]_
  # Need to use `use_resource = True` to make this work correctly
  # within tf.data.Dataset
  hist_counts = tf.get_local_variable(
      "hist-counts", n_hist_binds+1, tf.int64,
      tf.zeros_initializer(), use_resource=True)
  hist_bounds = tf.constant(hist_bounds, tf.int64)

  def bucket_fn(x):
    """Compute the element bucket and update the histogram."""
    ix = len_fn(x)
    if ix.dtype == tf.int32:
      ix = tf.to_int64(ix)
    elif ix.dtype != tf.int64:
      raise ValueError("Len function returned a non-int")

    adds_to_bins = tf.to_int64(tf.greater(hist_bounds, ix))
    # pad with a 1 for the "larger than all" bin
    adds_to_bins = tf.pad(adds_to_bins, [[0, 1]], constant_values=1)
    new_counts = tf.assign_add(hist_counts, adds_to_bins)
    bin_ix = n_hist_binds - tf.reduce_sum(adds_to_bins)

    # Computes the quantile based on the counts of the exammple's bucket
    bucket_ix = tf.floordiv(((n_buckets-1) * new_counts[bin_ix]), new_counts[-1])
    return bucket_ix

  def reduce_fn(_, x):
    return x.padded_batch(batch_size, x.output_shapes)

  return tf.contrib.data.group_by_window(bucket_fn, reduce_fn, batch_size)


def as_initialized_variable(x, var_name, local=True):
  """Build a variable the is initialized to `x`, but without adding
  `x` to the tensorflow graph.

  The main reason to do this is to avoid the tensorflow
  graph becoming bloating with huge constants, which can make some operation very slow.
  This is accomplished by `hiding` the variable behind a py_fun intitializer
  """
  init_fn = tf.py_func(lambda: x, [], tf.float32, False)
  init_fn.set_shape(x.shape)
  if local:
    return tf.get_local_variable(var_name, initializer=init_fn)
  else:
    return tf.get_variable(var_name, initializer=init_fn)


def mask_logits(vec, mask):
  """Mask `vec` in logspace by setting out of bounds elements to very negative values"""
  if mask is None:
    return vec

  if mask.dtype == tf.int32:
    # Assume `mask` holds sequence lengths
    if len(vec.shape) not in [2, 3]:
      raise ValueError("Can't use a length mask on tensor of rank>3")
    mask = tf.sequence_mask(mask, tf.shape(vec)[1], tf.float32)
  else:
    mask = tf.cast(mask, tf.float32)

  if len(mask.shape) == (len(vec.shape) - 1):
    mask = tf.expand_dims(mask, len(vec.shape) - 1)

  return vec * mask - (1 - mask) * 1E20


def get_best_span(span_logits, bound):
  """Get spans with highest start+end score and length <= `bound`

  :param span_logits: [batch, seq_len, 2] start/end scores
  :param bound: Max span len
  :return: [batch, 2] highest scoring spans
  """
  batch, time = get_shape_tuple(span_logits)[:2]

  # [batch, time, time], per-span scores
  scores = tf.expand_dims(span_logits[:, :, 0], 2) + tf.expand_dims(span_logits[:, :, 1], 1)
  scores = tf.reshape(scores, [batch, -1])  # flattened [batch, time*time]

  # Mask span beyond `bound`
  if bound is not None:
    bound = tf.minimum(tf.convert_to_tensor(bound, tf.int32), time)
  bound_mask = tf.matrix_band_part(
    tf.ones((time, time)), tf.to_int32(0), tf.to_int32(-1) if bound is None else bound)
  bound_mask = tf.reshape(bound_mask, [-1])

  scores += tf.expand_dims(tf.log(bound_mask), 0)  # sets out-of-bounds span to -inf

  _, ix = tf.nn.top_k(scores, 1, False)  # get top spans
  ix = tf.squeeze(ix, 1)
  spans = tf.stack([ix // time, ix % time], 1)  # convert non-flattened indices
  return spans

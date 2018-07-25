import os
import sys
from functools import partial
import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.client import device_lib
from tensorflow.contrib.crf import viterbi_decode
from tqdm import tqdm
from sklearn.utils import shuffle

from finetune import config


def concat_or_stack(tensors, axis=0):
    try:
        return tf.concat(tensors, axis=axis)
    except ValueError:
        # tensors are scalars
        return tf.stack(tensors, axis=axis)


def shuffle_data(*args):
    """
    Thin passthrough fn to sklearn.utils.shuffle, but allows for passing through None values
    """
    shuffled = shuffle(arg for arg in args if arg is not None)
    results = []
    idx = 0
    for arg in args:
        if arg is None:
            results.append(arg)
        else:
            results.append(shuffled[idx])
            idx += 1
    return tuple(results)


def format_gpu_string(num):
    return '/device:GPU:{}'.format(num)


def get_available_gpus(config):
    if config.visible_gpus is not None:
        return config.visible_gpus
    local_device_protos = device_lib.list_local_devices()
    config.visible_gpus = [
        int(x.name.split(':')[-1]) for x in local_device_protos
        if x.device_type == 'GPU'
    ]
    return config.visible_gpus


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def np_softmax(x, t=1, axis=-1):
    x = x / t
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f


def _identity_init(shape, dtype, partition_info, scale):
    n = shape[-1]
    w = np.eye(n) * scale
    if len([s for s in shape if s != 1]) == 2:
        w = w.reshape(shape)
    return w.astype(np.float32)


def identity_init(scale=1.0):
    return partial(_identity_init, scale=scale)


def _np_init(shape, dtype, partition_info, w):
    return w


def np_init(w):
    return partial(_np_init, w=w)


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag
            indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, np_softmax(trellis, axis=-1)


def guarantee_initialized_variables(sess, keys=None):
    """
    Adapted from: https://stackoverflow.com/a/43601894
    """
    if keys is None:
        keys = [""]

    matching_vars = set()
    for key in keys:
        matching_vars |= set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, ".*{}.*".format(key)))
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in matching_vars])
    uninitialized_vars = [v for (v, f) in zip(matching_vars, is_not_initialized) if not f]
    if len(uninitialized_vars):
        sess.run(tf.variables_initializer(uninitialized_vars))


def find_trainable_variables(key, exclude=None):
    """
    Simple helper function to get trainable variables that contain a certain string in their name :param key:, whilst
    excluding variables whos full name (including scope) includes a substring :param excludes:.
    """
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))
    if exclude is not None:
        trainable_variables = [
            var for var in trainable_variables
            if not exclude in var.name
        ]
    return trainable_variables


def soft_split(*xs, n_splits=None):
    """
    Similar to tf.split but can accommodate batches that are not evenly divisible by n_splits.

    Useful for data parallelism across multiple devices, where the batch size is not necessarily divisible by the
    number of devices or is variable between batches.
    """
    if not n_splits or not isinstance(n_splits, int):
        raise ValueError("n_splits must be a valid integer.")

    x = xs[0]
    current_batch_size = shape_list(x)[0]
    n_per = tf.to_int32(tf.ceil(current_batch_size / n_splits))
    for i in range(n_splits):
        start = tf.minimum(i * n_per, current_batch_size)
        end = tf.minimum((i + 1) * n_per, current_batch_size)
        i_range = tf.range(start, end)
        yield [tf.gather(x, i_range) for x in xs]


def flatten(outer):
    return [el for inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


def list_transpose(l):
    return [list(i) for i in zip(*l)]


def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf"), tqdm_desc=None):
    n = len(datas[0])
    if truncate:
        n = (n // n_batch) * n_batch
    n = min(n, max_batches * n_batch)
    n_batches = 0

    for i in tqdm(range(0, n, n_batch), total=n // n_batch, ncols=80, leave=False, disable=(not verbose),
                  desc=tqdm_desc):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i + n_batch]
        else:
            yield (d[i:i + n_batch] for d in datas)
        n_batches += 1


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """
    force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x


def assign_to_gpu(gpu=0, params_device="/device:CPU:0"):
    """
        A device assignment function to place all variables on :param params_device: and everything else on gpu
        number :param gpu:

        Useful for data parallelism across multiple GPUs.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return params_device
        else:
            return "/gpu:%d" % gpu

    return _assign


def sample_with_temperature(logits, temperature):
    """Either argmax or random sampling.
    Args:
      logits: a Tensor.
      temperature: a float  0.0=argmax 1.0=random
    Returns:
      a Tensor with one fewer dimension than logits.
    """
    if temperature == 0.0:
        # TF argmax doesn't handle >5 dimensions, so we reshape here.
        logits_shape = shape_list(logits)
        argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=1)
        return tf.reshape(argmax, logits_shape[:-1])
    else:
        assert temperature > 0.0
        reshaped_logits = (
                tf.reshape(logits, [-1, shape_list(logits)[-1]]) / temperature)
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices,
                             shape_list(logits)[:logits.get_shape().ndims - 1])
        return choices


def average_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def sequence_decode(logits, transition_matrix):
    """ A simple py_func wrapper around the Viterbi decode allowing it to be included in the tensorflow graph. """

    def _sequence_decode(logits, transition_matrix):
        all_predictions = []
        all_logits = []
        for logit in logits:
            viterbi_sequence, viterbi_logits = viterbi_decode(logit, transition_matrix)
            all_predictions.append(viterbi_sequence)
            all_logits.append(viterbi_logits)
        return np.array(all_predictions, dtype=np.int32), np.array(all_logits, dtype=np.float32)

    return tf.py_func(_sequence_decode, [logits, transition_matrix], [tf.int32, tf.float32])


def finetune_to_indico_sequence(raw_texts, subseqs, labels, none_value=config.PAD_TOKEN):
    """
    Maps from the labeled substring format into the 'indico' format. This is the exact inverse operation to
    :meth indico_to_finetune_sequence:.

    The indico format is as follows:
        Raw text for X,
        Labels as a list of dicts, with each dict in the form:
        {
            'start': <Character index of the start of the labeled sequence>,
            'end': <Character index of the end of the labeled sequence>,
            'label': <A categorical label (int or string) that represents the category of the subsequence,
            'text': <Optionally, a field with the subsequence contained between the start and end.
        }

    The Labeled substring, or finetune internal, format is as follows.
    Each item of the data is a list strings of the form:
        ["The quick brown", "fox", "jumped over the lazy", ...]
    With the corresponding labels:
        ["PAD", "animal", "PAD", ...]

    It is the :param none_value: that is used to populate the PAD labels.
    :param data: A list of segmented text of the form list(list(str))
    :param labels: Categorical labels for each sub-string in data.
    :param none_value: The none value used to encode the input format.
    :return: Texts, annoatations both in the 'indico' format.
    """
    texts = []
    annotations = []
    for raw_text, doc_seq, label_seq in zip(raw_texts, subseqs, labels):
        doc_text = ""
        doc_annotations = []
        char_loc = 0
        for sub_str, label in zip(doc_seq, label_seq):
            stripped_text = sub_str.strip()
            doc_location = raw_text.find(stripped_text, char_loc)
            if label != none_value:
                doc_annotations.append(
                    {
                        "start": doc_location,
                        "end": doc_location + len(stripped_text),
                        "label": label,
                        "text": stripped_text
                    }
                )
            char_loc = doc_location + len(stripped_text)
        annotations.append(doc_annotations)
    return raw_texts, annotations


def indico_to_finetune_sequence(texts, labels=None, none_value=config.PAD_TOKEN):
    """
    Maps from the 'indico' format sequence labeling data. Into a labeled substring format. This is the exact inverse of
    :meth finetune_to_indico_sequence:.

    The indico format is as follows:
        Raw text for X,
        Labels as a list of dicts, with each dict in the form:
        {
            'start': <Character index of the start of the labeled sequence>,
            'end': <Character index of the end of the labeled sequence>,
            'label': <A categorical label (int or string) that represents the category of the subsequence,
            'text': <A field containing the sub-sequence contained between the start and end.
        }

    The Labeled substring, or finetune internal, format is as follows.
    Each item of the data is a list strings of the form:
        ["The quick brown", "fox", "jumped over the lazy", ...]
    With the corresponding labels:
        ["PAD", "animal", "PAD", ...]

    It is the :param none_value: that is used to populate the PAD labels.

    :param texts: A list of raw text.
    :param labels: A list of targets of the form list(list(dict))).
    :param none_value: A categorical label to use as the none value.
    :return: Segmented Text, Labels of the form described above.
    """
    all_subseqs = []
    all_labels = []

    # placeholder for inference time
    if labels is None:
        labels = [[]] * len(texts)

    for text, label_seq in zip(texts, labels):
        last_loc = 0
        doc_subseqs = []
        doc_labels = []
        for annotation in label_seq:
            start = annotation["start"]
            end = annotation["end"]
            label = annotation["label"]
            if start != last_loc:
                doc_subseqs.append(text[last_loc:start])
                doc_labels.append(none_value)
            doc_subseqs.append(text[start:end])
            doc_labels.append(label)
            last_loc = end
        doc_subseqs.append(text[last_loc:])
        doc_labels.append(none_value)
        all_subseqs.append(doc_subseqs)
        all_labels.append(doc_labels)
    return all_subseqs, all_labels

import os
import sys
from functools import partial

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


def np_softmax(x, t=1):
    x = x / t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


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


def find_trainable_variables(key, exclude=None):
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))
    if exclude is not None:
        trainable_variables = [
            var for var in trainable_variables
            if not exclude in var.name 
        ]
    return trainable_variables


def soft_split(*xs, n_splits=None):
    """
    Similar to tf.split but can accomodate batches that are not evenly divisible by n_splits
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


def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf"), tqdm_desc=None):
    n = len(datas[0])
    if truncate:
        n = (n // n_batch) * n_batch
    n = min(n, max_batches * n_batch)
    n_batches = 0
    
    for i in tqdm(range(0, n, n_batch), total=n // n_batch, ncols=80, leave=False, disable=(not verbose), desc=tqdm_desc):
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
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x


def assign_to_gpu(gpu=0, params_device="/device:CPU:0"):
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

    def _sequence_decode(logits, transition_matrix):
        predictions = []
        scores = []
        for logit in logits:
            viterbi_sequence, viterbi_score = viterbi_decode(logit, transition_matrix)
            predictions.append(viterbi_sequence)
            scores.append(viterbi_score)
        return np.array(predictions, dtype=np.int32), np.array(scores, dtype=np.float32)

    return tf.py_func(_sequence_decode, [logits, transition_matrix], [tf.int32, tf.float32])
        

def finetune_to_indico_sequence(raw_texts, subseqs, labels, none_value=config.PAD_TOKEN):
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
            doc_subseqs.append(text[start: end])
            doc_labels.append(label)
            last_loc = end
        doc_subseqs.append(text[last_loc:])
        doc_labels.append(none_value)
        all_subseqs.append(doc_subseqs)
        all_labels.append(doc_labels)
    return all_subseqs, all_labels

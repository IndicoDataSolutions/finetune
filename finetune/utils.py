import os
import warnings
import numpy as np
import tensorflow as tf
from scipy import interpolate

from finetune.encoding import NLP
from finetune import config

def merge_leading_dims(X, target_rank):
    shape = [-1] + X.get_shape().as_list()[1 - target_rank:]
    return tf.reshape(X, shape)


def interpolate_pos_embed(positional_embed, new_len):
    xx = np.linspace(0, 512, new_len)
    newKernel = interpolate.RectBivariateSpline(np.arange(positional_embed.shape[0]),
                                                np.arange(positional_embed.shape[1]), positional_embed)
    return newKernel(xx, np.arange(positional_embed.shape[1]))


def concat_or_stack(tensors, axis=0):
    try:
        return tf.concat(tensors, axis=axis)
    except ValueError:
        # tensors are scalars
        return tf.stack(tensors, axis=axis)


def format_gpu_string(num):
    return '/device:GPU:{}'.format(num)


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
        yield [tf.gather(x, i_range) if x is not None else None for x in xs]


def flatten(outer):
    return [el for inner in outer for el in inner]


def list_transpose(l):
    return [list(i) for i in zip(*l)]


def sample_with_temperature(logits, temperature):
    """Either argmax or random sampling.
    Args:
      logits: a Tensor.
      temperature: a float  0.0=argmax 1.0=random
    Returns:
      a Tensor with one fewer dimension than logits.
    """
    logits_shape = shape_list(logits)
    if temperature == 0.0:
        return tf.argmax(logits, axis=-1)
    else:
        assert temperature > 0.0
        reshaped_logits = tf.reshape(logits, [-1, logits_shape[-1]]) / temperature
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices, logits_shape[:-1])
        return choices


def truncate_text(text, max_chars=100):
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def finetune_to_indico_sequence(raw_texts, subseqs, labels, probs=None, none_value=config.PAD_TOKEN,
                                subtoken_predictions=False):
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
    annotations = []
    for raw_text, doc_seq, label_seq, prob_seq in zip(raw_texts, subseqs, labels, probs or [None] * len(raw_texts)):
        tokens = NLP(raw_text)
        token_starts = [token.idx for token in tokens]
        token_ends = [token.idx + len(token.text) for token in tokens]
        n_tokens = len(tokens)

        doc_annotations = []
        annotation_ranges = set()
        start_idx = 0
        end_idx = 0
        raw_annotation_start = 0
        for sub_str, raw_label, confidences in zip(doc_seq, label_seq, prob_seq or [None] * len(doc_seq)):
            if not isinstance(raw_label, tuple):
                multi_label = False
                label_list = [raw_label]
            else:
                multi_label = True
                label_list = raw_label

            for label in label_list:

                stripped_text = sub_str.strip()

                raw_annotation_start = raw_text.find(stripped_text, raw_annotation_start)
                raw_annotation_end = raw_annotation_start + len(stripped_text)
                for i, item in enumerate(doc_annotations):
                    if item["label"] == label and raw_annotation_start - item["end"] <= 1:
                        raw_annotation_start = item["start"]
                        doc_annotations.pop(i)
                        break

                if raw_annotation_start == -1:
                    warnings.warn("Failed to find predicted sequence in text: {}.".format(
                        truncate_text(stripped_text)
                    ))
                    continue

                annotation_start = raw_annotation_start
                annotation_end = raw_annotation_end

                # if we don't want to allow subtoken predictions, adjust start and end to match
                # the start and ends of the nearest full tokens
                if not subtoken_predictions:
                    if multi_label:
                        start_idx = 0
                        end_idx = 0
                    if label != none_value:
                        # round to nearest token
                        while start_idx < n_tokens and annotation_start >= token_starts[start_idx]:
                            start_idx += 1
                        annotation_start = token_starts[start_idx - 1]
                        while end_idx < (n_tokens - 1) and annotation_end > token_ends[end_idx]:
                            end_idx += 1
                        annotation_end = token_ends[end_idx]

                text = raw_text[annotation_start:annotation_end]

                if label != none_value:
                    annotation = {
                        "start": annotation_start,
                        "end": annotation_end,
                        "label": label,
                        "text": text
                    }
                    if confidences is not None:
                        annotation["confidence"] = confidences

                    # prevent duplicate annotation edge case
                    if (annotation_start, annotation_end, label) not in annotation_ranges:
                        annotation_ranges.add((annotation_start, annotation_end, label))
                        doc_annotations.append(annotation)

        doc_annotations = sorted([dict(items) for items in doc_annotations], key=lambda x: x['start'])
        annotations.append(doc_annotations)
    return raw_texts, annotations


def indico_to_finetune_sequence(texts, labels=None, multi_label=True, none_value=config.PAD_TOKEN,
                                subtoken_labels=False):
    """
    Maps from the 'indico' format sequence labeling data. Into a labeled substring format. This is the exact inverse of
    :meth finetune_to_indico_sequence:.

    The indico format is as follows:
        Raw text for X,
        Labels as a list of dicts, with each dict in the form:
         labeled sequence>,
            'end': <Character index of the end of the labeled sequence>,
            'label': <A categorical label (int or string) that represents the category of the subsequence,
            'text': <A field containing the sub-sequence contained between the start and end.
        }

    The Labeled substring, or finetune internal, format is as follows.
    Each item of the data is a list strings of the form:{
            'start': <Character index of the start of the
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
        tokens = NLP(text)
        token_starts = [token.idx for token in tokens]
        token_ends = [token.idx + len(token.text) for token in tokens]
        n_tokens = len(tokens)

        label_seq = sorted(label_seq, key=lambda x: x["start"])
        last_loc = 0
        doc_subseqs = []
        doc_labels = []
        for i, annotation in enumerate(label_seq):
            start = annotation["start"]
            end = annotation["end"]
            label = annotation["label"]
            annotation_text = annotation.get("text")

            if annotation_text is not None and text[start:end] != annotation_text:
                raise ValueError(
                    "Annotation text does not match text specified by `start` and `end` indexes. "
                    "Text provided: `{}`.  Text extracted: `{}`.".format(
                        annotation_text,
                        text[start:end]
                    )
                )

            if not subtoken_labels:
                if label != none_value:
                    # round to nearest token
                    while start > 0 and start not in token_starts:
                        start -= 1
                    while end < len(text) and end not in token_ends:
                        end += 1

            if start > last_loc:
                doc_subseqs.append(text[last_loc:start])
                if multi_label:
                    doc_labels.append([none_value])
                else:
                    doc_labels.append(none_value)

            j = len(doc_labels) - 1
            split_dist = last_loc - end
            skip_end = 0
            if split_dist > 0:
                j -= 1
                if len(doc_subseqs[-1]) != split_dist:
                    dual_label_sub_seq = doc_subseqs[-1][-split_dist:]
                    doc_subseqs[-1] = doc_subseqs[-1][:-split_dist]
                    doc_subseqs.append(dual_label_sub_seq)
                    doc_labels.append(doc_labels[-1][:])
                    j -= 1

                skip_end = len(doc_subseqs[-1])

            if start < last_loc - skip_end:
                if not multi_label:
                    raise ValueError("Overlapping annotations requires the multi-label model")
                else:
                    split_dist = last_loc - start - skip_end
                    while split_dist >= len(doc_subseqs[j]):
                        doc_labels[j].append(label)
                        split_dist -= len(doc_subseqs[j])
                        j -= 1

                    if split_dist > 0:
                        dual_label_sub_seq = doc_subseqs[j][-split_dist:]
                        doc_subseqs[j] = doc_subseqs[j][:-split_dist]
                        doc_subseqs.insert(j + 1, dual_label_sub_seq)
                        doc_labels.insert(j + 1, doc_labels[j][:] + [label])

                    start = last_loc
            if start >= end:
                # degenerate label
                last_loc = max(start, end)
                continue

            doc_subseqs.append(text[start:end])
            if multi_label:
                doc_labels.append([label])
            else:
                doc_labels.append(label)

            last_loc = end

        if last_loc != len(text):
            doc_subseqs.append(text[last_loc:])
            if multi_label:
                doc_labels.append([none_value])
            else:
                doc_labels.append(none_value)
        all_subseqs.append(doc_subseqs)
        all_labels.append(doc_labels)
    return all_subseqs, all_labels

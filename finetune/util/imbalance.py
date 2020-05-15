"""
Utilities for dealing with class imbalance
"""
import tensorflow as tf
import numpy as np
from collections import Counter

from finetune.encoding.target_encoders import LabelEncoder
from finetune.errors import FinetuneError


def compute_class_weights(class_weights, class_counts, n_total=None, multilabel=False):
    """
    Optionally compute class weights based on target distribution
    """
    if class_weights is None:
        return

    options = {"linear", "log", "sqrt"}

    if isinstance(class_weights, str) and class_weights in options:
        counts = class_counts
        max_count = max(counts.values())
        computed_ratios = _compute_ratios(counts, n_total, multilabel=multilabel)
        computed_weights = {}
        for class_name, ratio in computed_ratios.items():
            if class_weights == "linear":
                computed_weights[class_name] = ratio
            elif class_weights == "sqrt":
                computed_weights[class_name] = np.sqrt(ratio)
            elif class_weights == "log":
                computed_weights[class_name] = np.log(ratio) + 1
        class_weights = computed_weights

    if not isinstance(class_weights, dict):
        raise FinetuneError(
            "Invalid value for config.class_weights: {}. "
            "Expected dictionary mapping from class name to weight or one of {}".format(
                class_weights, list(options)
            )
        )
    return class_weights


def _compute_ratios(counts, n_total, multilabel=False):
    computed_ratios = {}
    max_count = max(counts.values())
    for class_name, count in counts.items():
        if multilabel:
            ratio = (n_total - count) / count
        else:
            ratio = ratio = max_count / count
        computed_ratios[class_name] = ratio
    return computed_ratios 


def class_weight_tensor(class_weights, target_dim, label_encoder):
    """
    Convert from dictionary of class weights to tf tensor
    """
    class_weight_arr = np.ones(target_dim, dtype=np.float32)
    for i, cls in enumerate(label_encoder.target_labels):
        class_weight_arr[i] = class_weights.get(cls, 1.0)

    class_weight_tensor = tf.convert_to_tensor(value=class_weight_arr)
    return class_weight_tensor

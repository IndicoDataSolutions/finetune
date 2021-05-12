from copy import deepcopy
from collections import namedtuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from finetune.util.metrics import (
    _convert_to_token_list, _get_unique_classes,
    calc_precision, calc_recall, calc_f1,
)

Counts = namedtuple("TP", "FP", "FN")

def group_exact_counts(pred, label):
    """
    Return TP if groups match exactly
    """
    if (pred["label"] != label["label"] or
        len(pred["tokens"]) != len(label["tokens"])):
        return Counts(0, 1, 1)
    pred_spans = sorted(pred["tokens"], key=lambda x: x["start"])
    label_spans = sorted(label["tokens"], key=lambda x: x["start"])
    for pred_span, label_span in zip(pred_spans, label_spans):
        if (pred_span["start"] != label_span["start"] or
            pred_span["end"] != label_span["end"]):
            return Counts(0, 1, 1)
    return Counts(1, 0, 0)

def group_token_counts(pred, label):
    """
    Return TP, FP, FN counts based on tokens in groups
    """
    pred_tokens = _convert_to_token_list(pred["tokens"])
    label_tokens = _convert_to_token_list(label["tokens"])

    if pred["label"] != label["label"]:
        # All predicted tokens are false positives
        # All ground truth tokens are false negatives
        return Counts(0, len(pred_tokens), len(label_tokens))

    TP, FP, FN = 0, 0, 0
    # TP and FN
    for l_token in label_tokens:
        for p_token in pred_tokens:
            if (p_token["start"] == l_token["start"] and
                p_token["end"] == l_token["end"]):
                TP += 1
                break
        else:
            FN += 1
    # FP
    for p_token in pred_tokens:
        for l_token in label_tokens:
            if (p_token["start"] == l_token["start"] and
                p_token["end"] == l_token["end"]):
                break
        else:
            FP += 1
    return Counts(TP, FP, FN)

def group_overlap_counts(pred, label):
    """
    Return TP if groups overlap at all
    """
    if (pred["label"] != label["label"]):
        return Counts(0, 1, 1)
    for pred_span in pred["tokens"]:
        for label_span in label["tokens"]:
            start = max(pred_span["start"], label_span["start"])
            end = min(pred_span["end"], label_span["end"])
            # If overlapping...
            if start < end:
                return Counts(1, 0, 0)
    return Counts(0, 1, 1)

def group_superset_counts(pred, label):
    """
    Return TP if all label spans appear within pred spans
    """
    if (pred["label"] != label["label"]):
        return Counts(0, 1, 1)
    for label_span in label["tokens"]:
        for pred_span in pred["tokens"]:
            if (pred_span["start"] <= label_span["start"] and
                pred_span["end"] >= label_span["end"]):
                break
        else:
            return Counts(0, 1, 1)
    return Counts(1, 0, 0)

def joint_exact_counts(pred, label):
    """
    Return TP if joint groups match exactly
    """
    if (pred["label"] != label["label"] or
        len(pred["entities"]) != len(label["entities"])):
        return Counts(0, 1, 1)
    pred_entities = sorted(pred["entities"], key=lambda x: x["start"])
    label_entities = sorted(label["entities"], key=lambda x: x["start"])
    for pred_entity, label_entity in zip(pred_entities, label_entities):
        if (pred_entity["start"] != label_entity["start"] or
            pred_entity["end"] != label_entity["end"] or
            pred_entity["label"] != label_entity["label"]):
            return Counts(0, 1, 1)
    return Counts(1, 0, 0)

def joint_token_counts(pred, label):
    """
    Return TP, FP, FN counts based on tokens in joint groups
    """
    pred_tokens = _convert_to_token_list(pred["entities"])
    label_tokens = _convert_to_token_list(label["entities"])

    if pred["label"] != label["label"]:
        # All predicted tokens are false positives
        # All ground truth tokens are false negatives
        return Counts(0, len(pred_tokens), len(label_tokens))

    TP, FP, FN = 0, 0, 0
    # TP and FN
    for l_token in label_tokens:
        for p_token in pred_tokens:
            if (p_token["start"] == l_token["start"] and
                p_token["end"] == l_token["end"] and 
                p_token["label"] == l_token["label"]):
                TP += 1
                break
        else:
            FN += 1
    # FP
    for p_token in pred_tokens:
        for l_token in label_tokens:
            if (p_token["start"] == l_token["start"] and
                p_token["end"] == l_token["end"] and 
                p_token["label"] == l_token["label"]):
                break
        else:
            FP += 1
    return Counts(TP, FP, FN)

def joint_overlap_counts(pred, label):
    """
    Return TP if joint groups overlap at all
    """
    if (pred["label"] != label["label"]):
        return Counts(0, 1, 1)
    for pred_entity in pred["entities"]:
        for label_entity in label["entities"]:
            start = max(pred_entity["start"], label_entity["start"])
            end = min(pred_entity["end"], label_entity["end"])
            # If overlapping and same label...
            if start < end and pred_entity["label"] == label_entity["label"]:
                return Counts(1, 0, 0)
    return Counts(0, 1, 1)

def joint_superset_counts(pred, label):
    """
    Return TP if all label entities appear within pred entities
    """
    if (pred["label"] != label["label"]):
        return Counts(0, 1, 1)
    for label_entity in label["entities"]:
        for pred_entity in pred["entities"]:
            if (pred_entity["start"] <= label_entity["start"] and
                pred_entity["end"] >= label_entity["end"] and
                pred_entity["label"] == label_entity["label"]):
                break
        else:
            return Counts(0, 1, 1)
    return Counts(1, 0, 0)

def get_count_fn(metric_type, span_type):
    """
    Returns a function used to count TPs, FPs and FNs between two groups

    Group count functions compare based on the label of the group and the token
    spans contained within the group.

    Joint count functions compare based on the label of the group and the
    entity spans contained within the group.
    """
    fns = {
        "group": {
            "exact": group_exact_counts,
            "token": group_token_counts,
            "overlap": group_overlap_counts,
            "superset": group_superset_counts,
        },
        "joint": {
            "exact": joint_exact_counts,
            "token": joint_token_counts,
            "overlap": joint_overlap_counts,
            "superset": joint_superset_counts,
        },
    }
    return fns[metric_type.lower()][span_type.lower()]

def calc_group_assignment(preds, labels, count_fn):
    """
    Calculate optimal pred to label group assignments
    """
    counts = [[(0, 0, 0) for _ in len(labels)] for _ in len(preds)]
    costs = [[0 for _ in len(labels)] for _ in len(preds)]

    for i, pred in enumerate(preds):
        for j, label in enumerate(labels):
            TP, FP, FN = count_fn(pred, label)
            counts[i][j] = (TP, FP, FN)
            costs[i][j] =  -(TP - (FP + FN) / 2)

    pred_idxs, label_idxs = linear_sum_assignment(costs)
    pred_idxs, label_idxs = list(pred_idxs), list(label_idxs)

    return list(zip(pred_idxs, label_idxs)), counts

def calc_class_counts(all_preds, all_labels, metric_type="group", span_type="exact"):
    """
    Calculate per-class TP, FP and FN counts.
    Returns a dictionary of the form:
    {
        "class1": {
            "true_positives: 1,
            "false_positives: 1,
            "false_negatives": 1,
        }
        .
        .
        .
        "class_n: {...},
    }
    """
    count_fn = get_count_fn(metric_type, span_type)
    classes = _get_unique_classes(preds, labels)
    class_counts = {
        cls: {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        } for cls in classes
    }

    for doc_preds, doc_labels in zip(all_preds, all_labels):
        if doc_preds and doc_labels:
            assign_idxs, doc_counts = calc_group_assignment(
                doc_preds, doc_labels, count_fn
            )
            for pred_idx, label_idx in assign_idxs:
                TP, FP, FN = doc_counts[pred_idx][label_idx]
                pred_label = doc_preds[pred_idx]["label"]
                label_label = doc_labels[label_idx]["label"]

                class_counts[label_label]["true_positives"] += TP
                # False positives attributed to the predicted label
                class_counts[pred_label]["false_positives"] += FP
                # False negatives attributed to the ground truth label
                class_counts[label_label]["false_negatives"] += FN

        pred_idxs, label_idxs = list(zip(*assign_idxs))
        if len(doc_preds) > len(doc_labels):
            missing_idxs = set(range(len(doc_preds))) - set(pred_idxs)
            extra_groups = doc_preds
            # All examples in extra pred groups are false positives
            add_count = "false_positives"
        elif len(label_groups) > len(pred_groups):
            missing_idxs = set(range(len(doc_labels))) - set(label_idxs)
            extra_groups = doc_labels
            # All examples in extra label groups are false negatives
            add_count = "false_negatives"
        for idx in missing_idxs:
            group = extra_groups[idx]
            if span_type == "token" and metric_type == "group":
                group_len = len(_convert_to_token_list(group["tokens"]))
            elif span_type == "token" and metric_type == "joint":
                group_len = len(_convert_to_token_list(group["entities"]))
            else:
                group_len = 1
            class_counts[group["label"]][add_count] += group_len

    return class_counts

def calc_class_metrics(preds, labels, metric_type="group", span_type="exact"):
    """
    Calculates a set of metrics for each class.
    Returns a dictioary of the form:
    {
        "class1": {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": 2,
            "true_positives: 1,
            "false_positives: 1,
            "false_negatives": 1,
        }
        .
        .
        .
        "class_n: {...},
    }
    """
    class_counts = calc_class_counts(
        preds, labels, metric_type=metric_type, span_type=span_type
    )
    per_class_metrics = {}
    for cls, counts in class_counts.items():
        TP = counts["true_positives"]
        FP = counts["false_positives"]
        FN = counts["false_negatives"]

        precision = calc_precision(TP, FP)
        recall = calc_recall(TP, FN)
        f1 = calc_f1(recall, precision)
        support = TP + FN

        per_class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
            "true_positives": TP,
            "false_positives": FP,
            "false_negatives": FN,
        }
    return per_class_metrics

def micro_avg(class_metrics):
    """
    Takes a dictionary of class metrics and calculates micro F1
    """
    TP = sum([c["true_positives"] for c in class_metrics.values()])
    FP = sum([c["false_positives"] for c in class_metrics.values()])
    FN = sum([c["false_negatives"] for c in class_metrics.values()])
    precision = calc_precision(TP, FP)
    recall = calc_recall(TP, FN)
    return calc_f1(recall, precision)

def macro_avg(class_metrics):
    """
    Takes a dictionary of class metrics and calculates macro F1
    """
    f1s = [c["f1_score"] for c in class_metrics.values()]
    return np.average(f1s)

def weighted_avg(class_metrics):
    """
    Takes a dictionary of class metrics and calculates weighted F1
    """
    classes = list(class_metrics.keys())
    f1s = [class_metrics[c]["f1_score"] for c in classes]
    supports = [class_metrics[c]["support"] for c in classes]
    return np.average(f1s, weight=supports)

def get_average(average, class_metrics):
    fns = {
        "micro": micro_avg,
        "macro": macro_avg,
        "weighted": weighted_avg
    }
    return fns[average.lower()](class_metrics)

def group_metrics(preds, labels, span_type="exact", average=None):
    """
    Takes a set of group predictions and labels and returns either per-class
    metrics or average metrics.

    Note: Takes lists of groups. Output from grouping models should be unzipped
    to isolate group and NER predictions before being evaluated.
    """
    per_class_metrics = calc_class_metrics(
        preds, labels, metric_type="group", span_type=span_type
    )
    if average:
        # Returns single F1 value
        return get_average(average, per_class_metrics)
    else:
        return per_class_metrics

def entity_in_group(entity, group):
    """
    Returns True if entity is contained in a group span, False otherwise
    """
    for span in group["tokens"]:
        if (span["start"] <= entity["start"] and
            span["end"] >= entity["end"]):
            return True
    return False

def attach_entitites(groups, entities):
    """
    Attaches a documents entities to the documents groups
    """
    for group in groups:
        group["entities"] = []
        for entity in entities:
            if entity_in_group(entity, group):
                group["entities"].append(entity)
    return groups

def joint_metrics(preds, labels, span_type="exact", average=None):
    """
    Takes a set of NER + group predictions and labels and returns either per-class
    metrics or average metrics.

    Note: Takes lists of NER and group spans. Output of grouping models can be
    directly used.
    """
    pred_groups, label_groups = [], []
    # Attach entities to groups
    for doc_pred, doc_label in zip(preds, labels):
        # Copy so we don't modify input groups
        doc_pred_entities, doc_pred_groups = doc_pred
        doc_pred_groups = deepcopy(doc_pred_groups)
        doc_label_entities, doc_label_groups = doc_label
        doc_label_groups = deepcopy(doc_label_groups)

        doc_pred_groups = attach_entities(doc_pred_entities, doc_pred_groups)
        doc_label_groups = attach_entities(doc_label_entites, doc_label_groups)

        # Only need groups
        pred_groups.append(doc_pred_groups)
        label_groups.append(doc_label_groups)

    per_class_metrics = calc_class_metrics(
        pred_groups, label_groups, metric_type="joint", span_type=span_type
    )
    if average:
        # Returns single F1 value
        return get_average(average, per_class_metrics)
    else:
        return per_class_metrics

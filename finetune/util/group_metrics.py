import numpy as np
from scipy.optimize import linear_sum_assignment

from finetune.util.metrics import (
    _convert_to_token_list, _get_unique_classes,
    calc_precision, calc_recall, calc_f1,
)

def group_exact_counts(pred, label):
    """
    Return TP if groups match exactly
    """
    if (pred["label"] != label["label"] or
        len(pred["tokens"]) != len(label["tokens"])):
        return {"TP": 0, "FP": 1, "FN": 1}
    pred_spans = sorted(pred["tokens"], key=lambda x: x["start"])
    label_spans = sorted(label["tokens"], key=lambda x: x["start"])
    for p, l in zip(pred_spans, label_spans):
        if (p["start"] != l["start"] or
            p["end"] != l["end"]):
            return {"TP": 0, "FP": 1, "FN": 1}
    return {"TP": 1, "FP": 0, "FN": 0}

def group_token_counts(pred, label):
    """
    Return TP, FP, FN counts based on tokens
    """
    pred_tokens = _convert_to_token_list(pred["tokens"])
    label_tokens = _convert_to_token_list(label["tokens"])

    if pred["label"] != label["label"]:
        # All predicted tokens are false positives
        # All ground truth tokens are false negatives
        return {
            "TP": 0, "FP": len(pred_tokens), "FN": len(label_tokens),
            "pred_len": len(pred_tokens),
            "label_len": len(label_tokens),
        }

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
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "pred_len": len(pred_tokens),
        "label_len": len(label_tokens),
    }

def group_overlap_counts(pred, label):
    """
    Return TP if groups overlap at all
    """
    if (pred["label"] != label["label"]):
        return {"TP": 0, "FP": 1, "FN": 1}
    for pred_span in pred["tokens"]:
        for label_span in label["tokens"]:
            start = max(pred_span["start"], label_span["start"])
            end = min(pred_span["end"], label_span["end"])
            # If overlapping...
            if start < end:
                return {"TP": 1, "FP": 0, "FN": 0}
    return {"TP": 0, "FP": 1, "FN": 1}

def group_superset_counts(pred, label):
    """
    Return TP if all label spans appear within pred spans
    """
    if (pred["label"] != label["label"]):
        return {"TP": 0, "FP": 1, "FN": 1}
    for label_span in label["tokens"]:
        for pred_span in pred["tokens"]:
            if (pred_span["start"] <= label_span["start"] and
                pred_span["end"] >= label_span["end"]):
                break
        else:
            return {"TP": 0, "FP": 1, "FN": 1}
    return {"TP": 1, "FP": 0, "FN": 0}

def get_count_fn(span_type):
    fns = {
        "exact": group_exact_counts,
        "token": group_token_counts,
        "overlap": group_overlap_counts,
        "superset": group_superset_counts,
    }
    return fns[span_type.lower()]

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
    
    pred_idxs = list(pred_idxs)
    label_idxs = list(label_idxs)
    return list(zip(pred_idxs, label_idxs)), counts

def calc_class_counts(all_preds, all_labels, span_type="exact"):
    """
    Calculate per-class TP, FP and FN counts
    """
    count_fn = get_count_fn(span_type)
    classes = _get_unique_classes(preds, labels)
    class_counts = {
        cls: {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        } for cls in classes
    }

    for doc_preds, doc_labels in zip(all_preds, all_labels):
        assign_idxs, doc_counts = calc_group_assignment(
            doc_preds, doc_labels, count_fn
        )
        for pred_idx, label_idx in assign_idxs:
            group_count = doc_counts[pred_idx][label_idx]
            pred_label = doc_preds[pred_idx]["label"]
            label_label = doc_labels[label_idx]["label"]
            class_counts[label_label]["true_positives"] += group_count["TP"]
            class_counts[pred_label]["false_positives"] += group_count["FP"]
            class_counts[label_label]["false_negatives"] += group_count["FN"]

        pred_idxs, label_idxs = list(zip(*assign_idxs))
        if len(doc_preds) > len(doc_labels):
            # All examples in extra pred groups are false positives
            pred_idxs = set(pred_idxs)
            all_idxs = set(range(len(doc_preds)))
            missing_idxs = all_idxs - pred_idxs
            for idx in missing_idxs:
                pred_label = doc_preds[idx]["label"]
                # Check if a group is multiple samples for token span type
                pred_len = doc_counts[idx][0].get("pred_len", 1)
                class_counts[pred_label]["false_positives"] += pred_len
        elif len(label_groups) > len(pred_groups):
            # All examples in extra label groups are false negatives
            label_idxs = set(label_idxs)
            all_idxs = set(range(len(doc_labels)))
            missing_idxs = all_idxs - label_idxs
            for idx in missing_idxs:
                label_label = doc_preds[idx]["label"]
                # Check if a group is multiple samples for token span type
                label_len = doc_counts[0][idx].get("label_len", 1)
                class_counts[label_label]["false_negatives"] += label_len

    return class_counts

def calc_class_metrics(class_counts):
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

def micro_avg(class_counts):
    classes = list(class_counts.keys())
    TP = sum([class_counts[c]["true_positives"] for c in classes])
    FP = sum([class_counts[c]["false_positives"] for c in classes])
    FN = sum([class_counts[c]["false_negatives"] for c in classes])
    precision = calc_precision(TP, FP)
    recall = calc_recall(TP, FN)
    return calc_f1(recall, precision)

def macro_avg(class_counts):
    f1s = [c["f1_score"] for _, c in class_counts.items()]
    return np.average(f1s)

def weighted_avg(class_counts):
    classes = list(class_counts.keys())
    f1s = [class_counts[c]["f1_score"] for c in classes]
    supports = [class_counts[c]["support"] for c in classes]
    return np.average(f1s, weight=supports)

def get_average_fn(average):
    fns = {
        "micro": micro_avg,
        "macro": macro_avg,
        "weighted": weighted_avg
    }
    return fns[average.lower()]

def group_metrics(preds, labels, span_type="exact", average=None):
    class_counts = calc_class_counts(preds, labels, span_type=span_type)
    per_class_metrics = calc_class_metrics(class_counts)

    if average:
        # Returns single F1 value
        average_fn = get_average_fn(average)
        return average_fn(per_class_metrics)
    else:
        return per_class_metrics

import os
import copy
from functools import partial
from collections import defaultdict, OrderedDict

import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary_iterator

import tabulate

from finetune.encoding.input_encoder import get_spacy

_EVENT_FILE_GLOB_PATTERN = "events.out.tfevents.*"


def _get_unique_classes(true, predicted):
    true_and_pred = list(true) + list(predicted)
    return list(set([seq["label"] for seqs in true_and_pred for seq in seqs]))


def _convert_to_token_list(annotations, doc_idx=None):
    nlp = get_spacy()
    tokens = []
    annotations = copy.deepcopy(annotations)

    for annotation in annotations:
        start_idx = annotation.get("start")
        tokens.extend(
            [
                {
                    "start": start_idx + token.idx,
                    "end": start_idx + token.idx + len(token.text),
                    "text": token.text,
                    "label": annotation.get("label"),
                    "doc_idx": doc_idx,
                }
                for token in nlp(annotation.get("text"))
            ]
        )

    return tokens


def sequence_labeling_token_confusion(text, true, predicted):
    nlp = get_spacy()
    none_class = "<None>"
    unique_classes = _get_unique_classes(true, predicted)
    unique_classes.append(none_class)

    true_per_token_all = []
    pred_per_token_all = []

    for i, (text_i, true_list, pred_list) in enumerate(zip(text, true, predicted)):
        tokens = nlp(text_i)
        true_per_token = []
        pred_per_token = []
        for token in tokens:
            token_start_end = {"start": token.idx, "end": token.idx + len(token.text)}
            for true_i in true_list:
                if sequences_overlap(token_start_end, true_i):
                    true_per_token.append(true_i["label"])
                    break
            else:
                true_per_token.append(none_class)

            for pred_i in pred_list:
                if sequences_overlap(token_start_end, pred_i):
                    pred_per_token.append(pred_i["label"])
                    break
            else:
                pred_per_token.append(none_class)
        true_per_token_all.extend(true_per_token)
        pred_per_token_all.extend(pred_per_token)
    cm = confusion_matrix(
        y_true=true_per_token_all, y_pred=pred_per_token_all, labels=unique_classes
    )
    return tabulate.tabulate(
        [["Predicted\nTrue", *unique_classes]]
        + [[l, *r] for l, r in zip(unique_classes, cm)]
    )


def sequence_labeling_token_counts(true, predicted):
    """
    Return FP, FN, and TP counts
    """

    unique_classes = _get_unique_classes(true, predicted)

    d = {
        cls_: {"false_positives": [], "false_negatives": [], "true_positives": []}
        for cls_ in unique_classes
    }

    for i, (true_list, pred_list) in enumerate(zip(true, predicted)):
        true_tokens = _convert_to_token_list(true_list, doc_idx=i)
        pred_tokens = _convert_to_token_list(pred_list, doc_idx=i)

        # correct + false negatives
        for true_token in true_tokens:
            for pred_token in pred_tokens:
                if (
                    pred_token["start"] == true_token["start"]
                    and pred_token["end"] == true_token["end"]
                ):

                    if pred_token["label"] == true_token["label"]:
                        d[true_token["label"]]["true_positives"].append(true_token)
                    else:
                        d[true_token["label"]]["false_negatives"].append(true_token)
                        d[pred_token["label"]]["false_positives"].append(pred_token)

                    break
            else:
                d[true_token["label"]]["false_negatives"].append(true_token)

        # false positives
        for pred_token in pred_tokens:
            for true_token in true_tokens:
                if (
                    pred_token["start"] == true_token["start"]
                    and pred_token["end"] == true_token["end"]
                ):
                    break
            else:
                d[pred_token["label"]]["false_positives"].append(pred_token)

    return d


def calc_recall(TP, FN):
    try:
        return TP / float(FN + TP)
    except ZeroDivisionError:
        return 0.0


def calc_precision(TP, FP):
    try:
        return TP / float(FP + TP)
    except ZeroDivisionError:
        return 0.0


def calc_f1(recall, precision):
    try:
        return 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        return 0.0


def seq_recall(true, predicted, span_type="token"):
    count_fn = get_seq_count_fn(span_type)
    class_counts = count_fn(true, predicted)
    results = {}
    for cls_, counts in class_counts.items():
        FN = len(counts["false_negatives"])
        TP = len(counts["true_positives"])
        results[cls_] = calc_recall(TP, FN)
    return results


def seq_precision(true, predicted, span_type="token"):
    count_fn = get_seq_count_fn(span_type)
    class_counts = count_fn(true, predicted)
    results = {}
    for cls_, counts in class_counts.items():
        FP = len(counts["false_positives"])
        TP = len(counts["true_positives"])
        results[cls_] = calc_precision(TP, FP)
    return results


def micro_f1(true, predicted, span_type="token"):
    count_fn = get_seq_count_fn(span_type)
    class_counts = count_fn(true, predicted)
    TP, FP, FN = 0, 0, 0
    for cls_, counts in class_counts.items():
        FN += len(counts["false_negatives"])
        TP += len(counts["true_positives"])
        FP += len(counts["false_positives"])
    recall = calc_recall(TP, FN)
    precision = calc_precision(TP, FP)
    return calc_f1(recall, precision)


def per_class_f1(true, predicted, span_type="token"):
    """
    F1-scores per class
    """
    count_fn = get_seq_count_fn(span_type)
    class_counts = count_fn(true, predicted)
    results = OrderedDict()
    for cls_, counts in class_counts.items():
        results[cls_] = {}
        FP = len(counts["false_positives"])
        FN = len(counts["false_negatives"])
        TP = len(counts["true_positives"])
        recall = calc_recall(TP, FN)
        precision = calc_precision(TP, FP)
        results[cls_]["support"] = FN + TP
        results[cls_]["f1-score"] = calc_f1(recall, precision)
    return results


def sequence_f1(true, predicted, span_type="token", average=None):
    """
    If average = None, return per-class F1 scores, otherwise
    return the requested model-level score.
    """
    if average == "micro":
        return micro_f1(true, predicted, span_type)

    f1s_by_class = per_class_f1(true, predicted, span_type)
    f1s = [value.get("f1-score") for key, value in f1s_by_class.items()]
    supports = [value.get("support") for key, value in f1s_by_class.items()]

    if average == "weighted":
        return np.average(np.array(f1s), weights=np.array(supports))
    if average == "macro":
        return np.average(f1s)
    else:
        return f1s_by_class


def strip_whitespace(y):
    label_text = y["text"]
    lstripped = label_text.lstrip()
    new_start = y["start"] + (len(label_text) - len(lstripped))
    stripped = label_text.strip()
    return {
        "text": label_text.strip(),
        "start": new_start,
        "end": new_start + len(stripped),
        "label": y["label"],
    }


def sequence_labeling_token_precision(true, predicted):
    """
    Token level precision
    """
    return seq_precision(true, predicted, span_type="token")


def sequence_labeling_token_recall(true, predicted):
    """
    Token level recall
    """
    return seq_recall(true, predicted, span_type="token")


def sequence_labeling_micro_token_f1(true, predicted):
    """
    Token level F1
    """
    return micro_f1(true, predicted, span_type="token")


def sequences_overlap(x: dict, y: dict) -> bool:
    return x["start"] < y["end"] and y["start"] < x["end"]


def sequence_exact_match(true_seq, pred_seq):
    """
    Boolean return value indicates whether or not seqs are exact match
    """
    true_seq = strip_whitespace(true_seq)
    pred_seq = strip_whitespace(pred_seq)
    return pred_seq["start"] == true_seq["start"] and pred_seq["end"] == true_seq["end"]


def sequence_superset(true_seq, pred_seq):
    """
    Boolean return value indicates whether or predicted seq is a superset of target
    """
    true_seq = strip_whitespace(true_seq)
    pred_seq = strip_whitespace(pred_seq)
    return pred_seq["start"] <= true_seq["start"] and pred_seq["end"] >= true_seq["end"]


def sequence_labeling_counts(true, predicted, equality_fn):
    """
    Return FP, FN, and TP counts
    """
    unique_classes = _get_unique_classes(true, predicted)

    d = {
        cls_: {"false_positives": [], "false_negatives": [], "true_positives": []}
        for cls_ in unique_classes
    }

    for i, (true_annotations, predicted_annotations) in enumerate(zip(true, predicted)):
        # add doc idx to make verification easier
        for annotations in [true_annotations, predicted_annotations]:
            for annotation in annotations:
                annotation["doc_idx"] = i

        for true_annotation in true_annotations:
            for pred_annotation in predicted_annotations:
                if equality_fn(true_annotation, pred_annotation):
                    if pred_annotation["label"] == true_annotation["label"]:
                        d[true_annotation["label"]]["true_positives"].append(
                            true_annotation
                        )
                        break
            else:
                d[true_annotation["label"]]["false_negatives"].append(true_annotation)

        for pred_annotation in predicted_annotations:
            for true_annotation in true_annotations:
                if (
                    equality_fn(true_annotation, pred_annotation)
                    and true_annotation["label"] == pred_annotation["label"]
                ):
                    break
            else:
                d[pred_annotation["label"]]["false_positives"].append(pred_annotation)

    return d


def get_seq_count_fn(span_type="token"):
    span_type_fn_mapping = {
        "token": sequence_labeling_token_counts,
        "overlap": partial(sequence_labeling_counts, equality_fn=sequences_overlap),
        "exact": partial(sequence_labeling_counts, equality_fn=sequence_exact_match),
        "superset": partial(sequence_labeling_counts, equality_fn=sequence_superset),
    }
    return span_type_fn_mapping[span_type]


def sequence_labeling_overlap_precision(true, predicted):
    """
    Sequence overlap precision
    """
    return seq_precision(true, predicted, span_type="overlap")


def sequence_labeling_overlap_recall(true, predicted):
    """
    Sequence overlap recall
    """
    return seq_recall(true, predicted, span_type="overlap")


def sequence_labeling_overlap_micro_f1(true, predicted):
    """
    Sequence overlap micro F1
    """
    return micro_f1(true, predicted, span_type="overlap")


def annotation_report(
    y_true,
    y_pred,
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    width=20,
):
    # Adaptation of https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/metrics/classification.py#L1363
    token_precision = sequence_labeling_token_precision(y_true, y_pred)
    token_recall = sequence_labeling_token_recall(y_true, y_pred)
    overlap_precision = sequence_labeling_overlap_precision(y_true, y_pred)
    overlap_recall = sequence_labeling_overlap_recall(y_true, y_pred)

    count_dict = defaultdict(int)
    for annotation_seq in y_true:
        for annotation in annotation_seq:
            count_dict[annotation["label"]] += 1

    seqs = [
        token_precision,
        token_recall,
        overlap_precision,
        overlap_recall,
        dict(count_dict),
    ]
    labels = set(token_precision.keys()) | set(token_recall.keys())
    target_names = [u"%s" % l for l in labels]
    counts = [count_dict.get(target_name, 0) for target_name in target_names]

    last_line_heading = "Weighted Summary"
    headers = [
        "token_precision",
        "token_recall",
        "overlap_precision",
        "overlap_recall",
        "support",
    ]
    head_fmt = u"{:>{width}s} " + u" {:>{width}}" * len(headers)
    report = head_fmt.format(u"", *headers, width=width)
    report += u"\n\n"
    row_fmt = u"{:>{width}s} " + u" {:>{width}.{digits}f}" * 4 + u" {:>{width}}" "\n"
    seqs = [[seq.get(target_name, 0.0) for target_name in target_names] for seq in seqs]
    rows = zip(target_names, *seqs)
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    report += u"\n"
    averages = [np.average(seq, weights=counts) for seq in seqs[:-1]] + [
        np.sum(seqs[-1])
    ]
    report += row_fmt.format(last_line_heading, *averages, width=width, digits=digits)
    return report


def read_eval_metrics(eval_dir):
    eval_metrics_dict = defaultdict(dict)
    if gfile.Exists(eval_dir):
        for event_file in gfile.Glob(os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
            for event in summary_iterator.summary_iterator(event_file):
                if not event.HasField("summary"):
                    continue
                metrics = {}
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        metrics[value.tag] = value.simple_value
                if metrics:
                    eval_metrics_dict[event.step].update(metrics)
    return eval_metrics_dict

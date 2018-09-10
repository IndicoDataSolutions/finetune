from collections import defaultdict

from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np

from finetune.encoding import NLP


def _convert_to_token_list(annotations, doc_idx=None):
    tokens = []

    for annotation in annotations:
        start_idx = annotation.get('start')
        tokens.extend([
            {
                'start': start_idx + token.idx,
                'end': start_idx + token.idx + len(token.text),
                'text': token.text,
                'label': annotation.get('label'),
                'doc_idx': doc_idx
            }
            for token in NLP(annotation.get('text'))
        ])

    return tokens


def sequence_labeling_token_counts(true, predicted):
    """
    Return FP, FN, and TP counts
    """

    unique_classes = set([seq['label'] for seqs in true for seq in seqs])

    d = {
        cls_: {
            'false_positives': [],
            'false_negatives': [],
            'correct': []
        }
        for cls_ in unique_classes
    }
    
    for i, (true_list, pred_list) in enumerate(zip(true, predicted)):
        true_tokens = _convert_to_token_list(true_list, doc_idx=i)
        pred_tokens = _convert_to_token_list(pred_list, doc_idx=i)

        # correct + false negatives
        for true_token in true_tokens:
            for pred_token in pred_tokens:
                if (pred_token['start'] == true_token['start'] and 
                    pred_token['end'] == true_token['end']):

                    if pred_token['label'] == true_token['label']:
                        d[true_token['label']]['correct'].append(true_token)
                    else:
                        d[true_token['label']]['false_negatives'].append(true_token)
                        d[pred_token['label']]['false_positives'].append(pred_token)
                    
                    break
            else:
                d[true_token['label']]['false_negatives'].append(true_token)

        # false positives
        for pred_token in pred_tokens:
            for true_token in true_tokens:
                if (pred_token['start'] == true_token['start'] and 
                    pred_token['end'] == true_token['end']):
                    break
            else:
                d[pred_token['label']]['false_positives'].append(pred_token)
    
    return d


def seq_recall(true, predicted, count_fn):
    class_counts = count_fn(true, predicted)
    results = {}
    for cls_, counts in class_counts.items():
        FN = len(counts['false_negatives'])
        TP = len(counts['correct'])
        try:
            results[cls_] = TP / float(FN + TP)
        except ZeroDivisionError: 
            results[cls_] = 0.
    return results


def seq_precision(true, predicted, count_fn):
    class_counts = count_fn(true, predicted)
    results = {}
    for cls_, counts in class_counts.items():
        FP = len(counts['false_positives'])
        TP = len(counts['correct'])
        try:
            results[cls_] = TP / float(FP + TP)
        except ZeroDivisionError:
            results[cls_] = 0.
    return results

def micro_f1(true, predicted, count_fn):
    class_counts = count_fn(true, predicted)
    TP, FP, FN = 0, 0, 0
    for cls_, counts in class_counts.items():
        FN += len(counts['false_negatives'])
        TP += len(counts['correct'])
        FP += len(counts['false_positives'])
    recall = TP/float(FN + TP)
    precision = TP / float(FP + TP)
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1

def sequence_labeling_token_precision(true, predicted):
    """
    Token level precision
    """
    return seq_precision(true, predicted, count_fn=sequence_labeling_token_counts)


def sequence_labeling_token_recall(true, predicted):
    """
    Token level recall
    """
    return seq_recall(true, predicted, count_fn=sequence_labeling_token_counts)

def sequence_labeling_micro_token_f1(true, predicted):
    """
    Token level F1
    """
    return micro_f1(true, predicted, count_fn=sequence_labeling_token_counts)

def sequences_overlap(true_seq, pred_seq):
    """
    Boolean return value indicates whether or not seqs overlap
    """
    start_contained = (pred_seq['start'] < true_seq['end'] and pred_seq['start'] >= true_seq['start'])
    end_contained = (pred_seq['end'] > true_seq['start'] and pred_seq['end'] <= true_seq['end'])
    return start_contained or end_contained


def sequence_labeling_overlaps(true, predicted):
    """
    Return FP, FN, and TP counts
    """
    unique_classes = set([annotation['label'] for annotations in true for annotation in annotations])

    d = {
        cls_: {
            'false_positives': [],
            'false_negatives': [],
            'correct': []
        }
        for cls_ in unique_classes
    }

    for i, (true_annotations, predicted_annotations) in enumerate(zip(true, predicted)):
        # add doc idx to make verification easier
        for annotations in [true_annotations, predicted_annotations]:
            for annotation in annotations:
                annotation['doc_idx'] = i
        
        for true_annotation in true_annotations:
            for pred_annotation in predicted_annotations:
                if sequences_overlap(true_annotation, pred_annotation):
                    if pred_annotation['label'] == true_annotation['label']:
                        d[true_annotation['label']]['correct'].append(true_annotation)
                    else:
                        d[true_annotation['label']]['false_negatives'].append(true_annotation)
                        d[pred_annotation['label']]['false_positives'].append(pred_annotation)
                    break
            else:
                d[true_annotation['label']]['false_negatives'].append(true_annotation)

        for pred_annotation in predicted_annotations:
            for true_annotation in true_annotations:
                if (sequences_overlap(true_annotation, pred_annotation) and
                        true_annotation['label'] == pred_annotation['label']):
                    break
            else:
                d[pred_annotation['label']]['false_positives'].append(pred_annotation)

    return d


def sequence_labeling_overlap_precision(true, predicted):
    """
    Sequence overlap precision
    """
    return seq_precision(true, predicted, count_fn=sequence_labeling_overlaps)


def sequence_labeling_overlap_recall(true, predicted):
    """
    Sequence overlap recall
    """
    return seq_recall(true, predicted, count_fn=sequence_labeling_overlaps)


def annotation_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, width=20):
    # Adaptation of https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/metrics/classification.py#L1363
    token_precision = sequence_labeling_token_precision(y_true, y_pred)
    token_recall = sequence_labeling_token_recall(y_true, y_pred)
    overlap_precision = sequence_labeling_overlap_precision(y_true, y_pred)
    overlap_recall = sequence_labeling_overlap_recall(y_true, y_pred)

    count_dict = defaultdict(int)
    for annotation_seq in y_true:
        for annotation in annotation_seq:
            count_dict[annotation['label']] += 1

    seqs = [token_precision, token_recall, overlap_precision, overlap_recall, dict(count_dict)]
    labels = set(token_precision.keys()) | set(token_recall.keys())
    target_names = [u'%s' % l for l in labels]
    counts = [count_dict.get(target_name) for target_name in target_names]

    last_line_heading = 'Weighted Summary'
    headers = ["token_precision", "token_recall", "overlap_precision", "overlap_recall", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>{width}}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'
    row_fmt = u'{:>{width}s} ' + u' {:>{width}.{digits}f}' * 4 + u' {:>{width}}' '\n'
    seqs = [
        [seq.get(target_name) for target_name in target_names]
        for seq in seqs
    ]
    rows = zip(target_names, *seqs)
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    report += u'\n'
    averages = [np.average(seq, weights=counts) for seq in seqs[:-1]] + [np.sum(seqs[-1])]
    report += row_fmt.format(last_line_heading, *averages, width=width, digits=digits)
    return report

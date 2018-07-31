from sklearn.metrics import accuracy_score, recall_score, precision_score

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


def sequence_labeling_counts(true, predicted):
    """
    Return FP, FN, and TP counts
    """
    d = {
        'false_positives': [],
        'false_negatives': [],
        'correct': []
    }
    
    for i, (true_list, pred_list) in enumerate(zip(true, predicted)):
        true_tokens = _convert_to_token_list(true_list, doc_idx=i)
        pred_tokens = _convert_to_token_list(pred_list, doc_idx=i)

        # correct + false negatives
        for true_token in true_tokens:
            for pred_token in pred_tokens:
                if (pred_token['start'] == true_token['start'] and 
                    pred_token['end'] == true_token['end']):
                    d['correct'].append(true_token)
                    break
            else:
                d['false_negatives'].append(true_token)

        # false positives
        for pred_token in pred_tokens:
            for true_token in true_tokens:
                if (pred_token['start'] == true_token['start'] and 
                    pred_token['end'] == true_token['end']):
                    break
            else:
                d['false_positives'].append(pred_token)
    
    return d


def sequence_labeling_recall(true, predicted):
    """
    Token level recall
    """
    counts = sequence_labeling_counts(true, predicted)
    FN = len(counts['false_negatives'])
    TP = len(counts['correct'])
    return TP / float(FN + TP)


def sequence_labeling_precision(true, predicted):
    """
    Token level precision
    """
    counts = sequence_labeling_counts(true, predicted)
    FP = len(counts['false_positives'])
    TP = len(counts['correct'])
    return TP / float(FP + TP)

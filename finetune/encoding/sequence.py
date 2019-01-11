from finetune.encoding import NLP
from finetune import config


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

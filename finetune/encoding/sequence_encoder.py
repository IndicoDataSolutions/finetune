import warnings

from finetune.util.logging import truncate_text


def assign_associations(labels, associations, none_value):
    idx_lookups = [{} for _ in labels]
    for i, (doc_label, doc_association) in enumerate(zip(labels, associations)):
        active_label_idx = -1
        for label, association in zip(doc_label, doc_association):
            if label == none_value:
                continue
            active_label_idx += 1
            for bpe_idx, _, _, _ in association:
                idx_lookups[i][bpe_idx] = active_label_idx

    all_candidates = []

    for idx_lookup, doc_label, doc_association in zip(idx_lookups, labels, associations):
        candidates = {}
        if doc_label == none_value:
            continue

        for association in doc_association:
            for bpe_idx, candidate_idx, candidate_label, candidate_prob in association:
                if candidate_label == none_value or candidate_idx not in idx_lookup:
                    continue

                if idx_lookup[bpe_idx] not in candidates:
                    candidates[idx_lookup[bpe_idx]] = []

                candidates[idx_lookup[bpe_idx]].append((idx_lookup[candidate_idx], candidate_label, candidate_prob))

        # TODO some how sample these candidates eg maximum probabilities, to fit some schema
        candidates = {k: max(v, key=lambda x: x[2]) for k, v in candidates.items()} # for now just pick maximum prob
        all_candidates.append(candidates)
    return all_candidates


def finetune_to_indico_sequence(raw_texts, subseqs, labels, encoder=None, probs=None, none_value=None,
                                subtoken_predictions=False, associations=None):
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
    if associations is not None:
        assoc_cleaned = assign_associations(labels, associations, none_value)
    else:
        assoc_cleaned = [None] * len(raw_texts)

    encoded_docs = encoder._encode(raw_texts)
    loop_vals = zip(raw_texts, subseqs, labels, probs or [None] * len(raw_texts), assoc_cleaned)
    for doc_idx, (raw_text, doc_seq, label_seq, prob_seq, associations_seq) in enumerate(loop_vals):
        tokens = encoded_docs.tokens[doc_idx]
        token_ends = encoded_docs.char_locs[doc_idx]
        token_lengths = [encoder._token_length(token) for token in tokens]
        token_starts = [end - length for end, length in zip(token_ends, token_lengths)]
        n_tokens = len(tokens)

        doc_annotations = []
        annotation_ranges = set()
        start_idx = 0
        end_idx = 0
        raw_annotation_start = 0
        for i, (sub_str, raw_label, confidences) in enumerate(zip(doc_seq, label_seq, prob_seq or [None] * len(doc_seq))):
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
                        "start": int(annotation_start),
                        "end": int(annotation_end),
                        "label": label,
                        "text": text
                    }
                    if associations_seq is not None and len(doc_annotations) in associations_seq:
                        index, relationship, prob = associations_seq[len(doc_annotations)]
                        annotation["associations"] = {
                            "index": index,
                            "relationship": relationship,
                            "prob": prob
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


def indico_to_finetune_sequence(texts, labels=None, encoder=None, multi_label=True, none_value=None,
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
    all_association_idx = []
    all_association_type = []
    all_idxs = []

    # placeholder for inference time
    if labels is None:
        labels = [[]] * len(texts)

    encoded_docs = encoder._encode(texts)

    for doc_idx, (text, label_seq) in enumerate(zip(texts, labels)):
        tokens = encoded_docs.tokens[doc_idx]
        token_ends = encoded_docs.char_locs[doc_idx]
        token_lengths = [encoder._token_length(token) for token in tokens]
        token_starts = [end - length for end, length in zip(token_ends, token_lengths)]
        n_tokens = len(tokens)

        label_seq = sorted(label_seq, key=lambda x: x["start"])
        last_loc = 0
        doc_subseqs = []
        doc_labels = []
        doc_association_idx = []
        doc_association_type = []
        doc_current_label_idx = []

        for i, annotation in enumerate(label_seq):
            start = annotation["start"]
            end = annotation["end"]
            label = annotation["label"]
            annotation_text = annotation.get("text")
            if "association" in annotation:
                association_idx = annotation["association"]["index"]
                association_type = annotation["association"]["relationship"]
            else:
                association_idx = -1
                association_type = none_value

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
                    doc_association_idx.append(-1)
                    doc_association_type.append(none_value)
                    doc_current_label_idx.append(-2)

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
                doc_association_idx.append(association_idx)
                doc_association_type.append(association_type)
                doc_current_label_idx.append(i)

            last_loc = end

        if last_loc != len(text):
            doc_subseqs.append(text[last_loc:])
            if multi_label:
                doc_labels.append([none_value])
            else:
                doc_labels.append(none_value)
                doc_association_idx.append(-1)
                doc_association_type.append(none_value)
                doc_current_label_idx.append(-2)

        all_subseqs.append(doc_subseqs)
        all_labels.append(doc_labels)
        all_association_idx.append(doc_association_idx)
        all_association_type.append(doc_association_type)
        all_idxs.append(doc_current_label_idx)

    return all_subseqs, all_labels, all_association_type, all_association_idx, all_idxs

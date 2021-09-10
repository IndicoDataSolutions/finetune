from abc import ABCMeta
import logging
import json

import numpy as np
from finetune.encoding.target_encoders import (
    BaseEncoder, 
    Seq2SeqLabelEncoder,
    SequenceLabelingEncoder,
)

LOGGER = logging.getLogger("finetune")

def try_decode(pred):
    try:
        return json.loads(pred)
    except:
        return []

class SequenceLabelingTextEncoder(Seq2SeqLabelEncoder):
    def transform(self, y):
        all_labels = []
        for y_i in y:
            # List of dictionaries, where the key is the label and the value is
            # the span being labeled

            # TODO: Add sorting of y_i before encoding, as Doc2Dict tends to do
            # bettter when labels and text appear in a consistent order

            labels = [{span["label"]: span["text"]} for span in y_i]
            all_labels.append(json.dumps(labels))
        ret = super().transform(all_labels)
        return ret

    def inverse_transform(self, preds, raw_texts):
        preds = super().inverse_transform(preds)
        preds = map(try_decode, preds)
        return self.decode_preds(preds, raw_texts)

    @staticmethod
    def decode_preds(preds, raw_texts):
        all_labels = []
        for raw_text, pred in zip(raw_texts, preds):
            if not type(pred) == list:
                all_labels.append([])
                continue
            doc_labels = []
            for json_label in pred:
                # json_label is a list of dictionaries of the form {key: text}
                if not type(json_label) == dict:
                    continue
                tag, text = list(json_label.items())[0]
                # Can't generate \n's, appear as a single n instead
                text = text.replace(' n ', ' \n ')
                label_start = raw_text.find(text)
                label_end = label_start + len(text)
                doc_labels.append({
                    "label": tag,
                    "start": label_start,
                    "end": label_end,
                    "text": text,
                })
            all_labels.append(doc_labels)
        return all_labels

class GroupLabelingTextEncoder(Seq2SeqLabelEncoder):
    def transform(self, y):
        all_labels = []
        for yi in y:
            doc_groups = []
            _, groups = yi
            for group in groups:
                # TODO: Add sorting of group tokens before encoding, as Doc2Dict tends to do
                # bettter when labels and text appear in a consistent order

                # List of strings, where each string is on of the spans of the group
                doc_groups.append([span["text"] for span in group["spans"]])
            all_labels.append(json.dumps(doc_groups))
        ret = super().transform(all_labels)
        return ret

    def inverse_transform(self, preds, raw_texts):
        preds = super().inverse_transform(preds)
        preds = map(try_decode, preds)
        return self.decode_preds(preds, raw_texts)
    
    @staticmethod
    def decode_preds(preds, raw_texts):
        all_groups = []
        for raw_text, pred in zip(raw_texts, preds):
            if not type(pred) == list:
                all_groups.append([])
                continue
            doc_groups = []
            for json_group in pred:
                if not type(json_group) == list:
                    continue
                # json_group is a lists of spans, represented as strings
                group_spans = []
                for span_text in json_group:
                    if not type(span_text) == str:
                        continue
                    # Can't generate \n's, appear as a single n instead
                    span_text = span_text.replace(' n ', ' \n ')
                    span_start = raw_text.find(span_text)
                    span_end = span_start + len(span_text)
                    group_spans.append({
                        "start": span_start,
                        "end": span_end,
                        "text": span_text,
                    })
                doc_groups.append({
                    "spans": group_spans,
                    "label": None
                })
            all_groups.append(doc_groups)
        return all_groups


class JointLabelingTextEncoder(Seq2SeqLabelEncoder):
    def transform(self, y):
        all_labels = []
        for yi in y:
            labels, groups = yi

            # TODO: Add sorting of labels and groups before encoding, as
            # Doc2Dict tends to do bettter when labels and text appear in a
            # consistent order

            # NER Labels
            doc_labels = [{span["label"]: span["text"]} for span in labels]
            # Group Labels
            doc_groups = []
            for group in groups:
                doc_groups.append([span["text"] for span in group["spans"]])
            # Combine
            all_labels.append(json.dumps([doc_labels, doc_groups]))
        ret = super().transform(all_labels)
        return ret
    
    def inverse_transform(self, preds, raw_texts):
        preds = super().inverse_transform(preds)

        preds = map(try_decode, preds)
        def split_decode(pred):
            # Ensure NER and group predictions were generated
            if len(pred) == 2:
                return pred
            else:
                return ([], [])
        preds = map(split_decode, preds)

        seq_preds, group_preds = list(zip(*preds))
        seq_ret = SequenceLabelingTextEncoder.decode_preds(seq_preds, raw_texts)
        group_ret = GroupLabelingTextEncoder.decode_preds(group_preds, raw_texts)
        return list(zip(seq_ret, group_ret))

def is_continuous(group):
    # If there is only a single span, the group is continuous
    return len(group["spans"]) == 1

class GroupSequenceLabelingEncoder(SequenceLabelingEncoder):
    """
    Produces encoded labels for a BIO tagging based grouping formulation.
    """
    def __init__(self, pad_token, bio_tagging=True):
        super().__init__(pad_token, bio_tagging=bio_tagging, group_tagging=True)

    def fit(self, labels):
        labels, groups = list(zip(*labels))
        super().fit(labels)
        
        # Add BG- and IG- version of labels
        n_classes = []
        for c in self.classes_:
            # Retain original label for entities outside of groups
            # (Equivalent to Outside (O) prefix)
            n_classes.append(c)
            n_classes.append("BG-" + c)
            n_classes.append("IG-" + c)
        self.classes_ = n_classes
        self.lookup = {c: i for i, c in enumerate(self.classes_)}
        
    def transform(self, out, labels):
        labels, groups = labels

        # Add a group_start value to every label for the sequence labeler
        # None -> Not in group, True -> Start of group, False -> In group
        # This could potentially be pulled out of the sequence labeler
        for label in labels:
            label["group_start"] = None
        for group in groups:
            if not is_continuous(group):
                continue

            # Gather entities inside group
            # Note that we assume groups are made up of only entities
            group_start = min([t["start"] for t in group["spans"]])
            group_end = max([t["end"] for t in group["spans"]])
            group_labels = []
            for label in labels:
                if label["start"] >= group_start and label["end"] <= group_end:
                    group_labels.append(label)
            # Add group_start information
            if group_labels:
                group_labels = sorted(group_labels, key=lambda x: x["start"])
                group_labels[0]["group_start"] = True
                for label in group_labels[1:]:
                    label["group_start"] = False

        ret = super().transform(out, labels)
        # Un-do the changes to the underlying data
        for label in labels:
            del label["group_start"]
        return ret

class MultiCRFGroupSequenceLabelingEncoder(SequenceLabelingEncoder):
    """
    Produces labels for a model with a seperate CRF for NER and group
    information.

    Outputs two label sequences, the first for the standard NER CRF and the
    second for the group CRF.
    """
    def __init__(self, pad_token, bio_tagging=True):
        super().__init__(pad_token, bio_tagging=bio_tagging, group_tagging=True)

    def fit(self, labels):
        labels, groups = list(zip(*labels))
        super().fit(labels)

        self.label_classes_ = self.classes_
        self.label_lookup = self.lookup

        self.group_classes_ = [self.pad_token, "BG-", "IG-"]
        self.group_lookup = {c: i for i, c in enumerate(self.group_classes_)}
        
    def transform(self, out, labels):
        labels, groups = labels

        # Create "fake" labels for groups
        group_labels = []
        for group in groups:
            if not is_continuous(group):
                continue
            group_start = min([t["start"] for t in group["spans"]])
            group_end = max([t["end"] for t in group["spans"]])
            group_text = " ".join([t["text"] for t in group["spans"]])
            group_labels.append({
                "start": group_start,
                "end": group_end,
                "label": "",
                "text": group_text,
                "group_start": True,
            })

        # Encode grouping labels
        self.classes_ = self.group_classes_
        self.lookup = self.group_lookup
        # Group tagging must be on and bio tagging must be off
        # (We use group tagging here instead of bio tagging for convenience, as
        # this way we don't have to convert back to group tags when decoding)
        self.group_tagging = True
        _bio, self.bio_tagging = self.bio_tagging, False
        encoded_group_labels =  super().transform(out, group_labels)

        # Encode NER labels
        self.classes_ = self.label_classes_
        self.lookup = self.label_lookup
        # Turn group tagging off and revert bio tagging
        self.group_tagging = False
        self.bio_tagging = _bio
        encoded_labels =  super().transform(out, labels)

        return [encoded_labels, encoded_group_labels]

    def inverse_transform(self, y, only_labels=False):
        labels, group_labels = y
        labels = [self.label_classes_[l] for l in labels]

        # Option to only return encoded labels for class weight calculations
        if only_labels:
            return labels

        group_labels = [self.group_classes_[l] for l in group_labels]
        # Replace pad tokens with empty prefixes
        group_labels = ["" if l == self.pad_token else l for l in group_labels]
        return [pre + tag for pre, tag in zip(group_labels, labels)]

class PipelineSequenceLabelingEncoder(SequenceLabelingEncoder):
    """
    Produces sequence labeling labels given group information.

    Parameters:
        group: When true, group information will be used for labels
    """
    def __init__(self, pad_token, group=True, bio_tagging=True):
        super().__init__(pad_token, bio_tagging=bio_tagging)
        self.group = group

    def fit(self, labels):
        if not self.group:
            labels, groups = list(zip(*labels))
            super().fit(labels)
        else:
            self.classes_ = [self.pad_token]
            if self.bio_tagging:
                self.classes_.extend(("B-GROUP", "I-GROUP"))
            else:
                self.classes_.append("GROUP")
            self.lookup = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, out, labels):
        labels, groups = labels
        if not self.group:
            return super().transform(out, labels)
        else:
            labels = []
            for group in groups:
                if not is_continuous(group):
                    continue
                group_start = min([t["start"] for t in group["spans"]])
                group_end = max([t["end"] for t in group["spans"]])
                group_text = " ".join([t["text"] for t in group["spans"]])
                labels.append({
                    "start": group_start,
                    "end": group_end,
                    "label": "GROUP",
                    "text": group_text,
                })
        return super().transform(out, labels)

class BROSEncoder(BaseEncoder):
    """
    Produces labels for BROS.
    
    First set of labels is a binary value indicating whether a token is the
    start of a group. Second set of labels is a index indicating what the next
    token in the group is.
    """
    def __init__(self, pad_token):
        self.classes_ = None
        self.pad_token = pad_token
        self.lookup = None

    def fit(self, labels):
        # STOP and CONT represent 0 and non-0 next tokens, respectively
        # STOP and CONT only matter when class weights are being used - see
        # BROSPipeline and util.imabalance.class_weight_tensor
        # Note: This means finetune's n_targets count used elsewhere is broken
        self.classes_ = [self.pad_token, "GROUP", "STOP", "CONT"]
        self.lookup = {c: i for i, c in enumerate(self.classes_)}

    def pre_process_label(self, out, labels):
        pad_idx = self.lookup[self.pad_token]
        return labels, pad_idx

    @staticmethod
    def group_overlaps(group, start, end, text, input_text):
        for span in group["spans"]:
            overlap, agree = SequenceLabelingEncoder.overlaps(span, start, end,
                                                              text, input_text)
            if overlap:
                return overlap, agree
        return False, False

    def transform(self, out, labels):
        input_text = "".join(out.input_text)
        labels, groups = labels
        labels, pad_idx = self.pre_process_label(out, labels)
        start_token_labels = [pad_idx for _ in out.tokens]
        # Note that the default next token is 0. This is safe, as 0 is the
        # start token which no token should normally point to
        next_token_labels = [0 for _ in out.tokens]

        for group in groups:
            current_tag = "GROUP"
            prev_idx = None
            group_end = max([g["end"] for g in group["spans"]])
            for i, (start, end, text) in enumerate(zip(out.token_starts, out.token_ends, out.tokens)):
                if group_end < (start + end + 1) // 2:
                    break
                overlap, agree = self.group_overlaps(group, start, end, text, input_text)
                if overlap:
                    if not agree:
                        raise ValueError("Tokens and labels do not align")
                    if not prev_idx and start_token_labels[i] != pad_idx:
                        LOGGER.warning("Overlapping start tokens found!")
                    if prev_idx and next_token_labels[i] != pad_idx:
                        LOGGER.warning("Overlapping next tokens found!")
                    if current_tag not in self.lookup:
                        LOGGER.warning(
                            "Attempting to encode unknown labels : {}, ignoring for now but this will likely not "
                            "result in desirable behaviour. Available labels are {}".format(current_tag, self.lookup.keys())
                        )
                    if prev_idx is None:
                        # Only the first token needs a start token label
                        start_token_labels[i] = self.lookup[current_tag]
                        prev_idx = i
                    elif prev_idx is not None:
                        next_token_labels[prev_idx] = i
                        prev_idx = i

        return [start_token_labels, next_token_labels]

    def inverse_transform(self, y):
        start_tokens, next_tokens = y
        # Placeholder for is we ever add labels
        start_tokens = [self.classes_[l] for l in start_tokens]
        return (start_tokens, next_tokens)

class GroupRelationEncoder(BROSEncoder):
    """
    Produces labels for a group relation model.
    
    Labels are of shape [n_groups, seq_len], where each element is a 0 or 1 and
    represents whether a token belongs to a certain group, and where the last
    group represents the pad group
    """
    def __init__(self, pad_token, n_groups):
        self.classes_ = None
        self.lookup = None
        self.pad_token = pad_token
        self.n_groups = n_groups

    def fit(self, labels):
        # PAD and GROUP represent tokens in the pad group and all other tokens
        # PAD and GROUP only matter when class weights are being used - see
        # GroupRelationsPipeline and util.imabalance.class_weight_tensor
        # Note: This means finetune's n_targets count used elsewhere is broken
        self.classes_ = ["GROUP", "PAD"]
        self.lookup = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, out, labels):
        input_text = "".join(out.input_text)
        labels, groups = labels

        if len(groups) > self.n_groups - 1:
            print(groups)
            raise ValueError(f"{len(groups)} groups found, more than n_groups!")

        # Build relation matrix
        encoded_labels = [[0 for _ in range(len(out.tokens))]
                          for _ in range(self.n_groups - 1)]
        # Final group is reserved for pad
        # Tokens default to this group
        encoded_labels.append([1 for _ in range(len(out.tokens))])
        for i, group in enumerate(groups):
            group_end = max([g["end"] for g in group["spans"]])
            for j, (start, end, text) in enumerate(zip(out.token_starts, out.token_ends, out.tokens)):
                if group_end < (start + end + 1) // 2:
                    break
                overlap, agree = self.group_overlaps(group, start, end, text, input_text)
                if overlap:
                    if not agree:
                        raise ValueError("Tokens and groups do not align")
                    encoded_labels[i][j] = 1
                    encoded_labels[-1][j] = 0

        return encoded_labels

    def inverse_transform(self, y):
        return y

class JointBROSEncoder(BROSEncoder, SequenceLabelingEncoder):
    """
    Produces labels for a joint BROS / Sequence Labeling model.
    
    First set of labels are standard NER sequence labeling models, second and
    third set of labels are BROS start and next token labels.
    """

    def __init__(self, pad_token, bio_tagging=False):
        # Can potentially replace these direct calls with super()s, but seems
        # like it'll be hard to maintain so leaving as is for now
        SequenceLabelingEncoder.__init__(self, pad_token,
                                         bio_tagging=bio_tagging)

    def fit(self, labels):
        labels, groups = list(zip(*labels))
        BROSEncoder.fit(self, labels)
        self.group_classes_, self.group_lookup = self.classes_, self.lookup
        SequenceLabelingEncoder.fit(self, labels)
        self.ner_classes_, self.ner_lookup = self.classes_, self.lookup
        
        # Create a combination of both classes so we can create a single class
        # weight tensor in util.imbalance.class_weight_tensor
        # Note: Finetune calculates n_labels as the length of classes_, so this
        # caues n_labels to be too large. We manually adjust for this in the
        # joint_bros target block
        self.classes_ = [c for c in self.ner_classes_]
        self.classes_.extend(self.group_classes_)

    def transform(self, out, labels):
        labels, groups = labels

        _classes_ = self.classes_

        self.classes_, self.lookup = self.group_classes_, self.group_lookup
        group_labels = BROSEncoder.transform(self, out, (labels, groups))
        start_token_labels, next_token_labels = group_labels

        self.classes_, self.lookup = self.ner_classes_, self.ner_lookup
        ner_labels = SequenceLabelingEncoder.transform(self, out, labels)

        # Ensure we always have the full list of classes for class weights
        self.classes_ = _classes_

        return [ner_labels, start_token_labels, next_token_labels]

    def inverse_transform(self, y, only_labels=False):
        tags, start_tokens, next_tokens = y

        _classes_ = self.classes_

        self.classes_, self.lookup = self.group_classes_, self.group_lookup
        group_tags = BROSEncoder.inverse_transform(self, (start_tokens, next_tokens))
        start_tokens, next_tokens = group_tags

        self.classes_, self.lookup = self.ner_classes_, self.ner_lookup
        tags = SequenceLabelingEncoder.inverse_transform(self, tags)

        self.classes_ = _classes_

        return (tags, start_tokens, next_tokens)

class JointGroupRelationEncoder(GroupRelationEncoder, SequenceLabelingEncoder):
    """
    Produces labels for a joint Group Relation / Sequence Labeling model.

    Sequence labels are packed into the group relation labels in the form of an
    additional group added to the end.
    """

    def __init__(self, pad_token, n_groups, bio_tagging=False):
        SequenceLabelingEncoder.__init__(self, pad_token,
                                         bio_tagging=bio_tagging)
        GroupRelationEncoder.__init__(self, pad_token, n_groups)

    def fit(self, labels):
        labels, groups = list(zip(*labels))
        GroupRelationEncoder.fit(self, labels)
        self.group_classes_, self.group_lookup = self.classes_, self.lookup
        SequenceLabelingEncoder.fit(self, labels)
        self.ner_classes_, self.ner_lookup = self.classes_, self.lookup
        
        # Create a combination of both classes so we can create a single class
        # weight tensor in util.imbalance.class_weight_tensor
        # Note: Finetune calculates n_labels as the length of classes_, so this
        # caues n_labels to be too large. We manually adjust for this in the
        # joint_group_relation target block
        self.classes_ = [c for c in self.ner_classes_]
        self.classes_.extend(self.group_classes_)

    def transform(self, out, labels):
        labels, groups = labels

        _classes_ = self.classes_

        self.classes_, self.lookup = self.group_classes_, self.group_lookup
        group_labels = GroupRelationEncoder.transform(self, out, (labels, groups))

        self.classes_, self.lookup = self.ner_classes_, self.ner_lookup
        ner_labels = SequenceLabelingEncoder.transform(self, out, labels)

        self.classes_ = _classes_

        return {"groups": group_labels, "tags": ner_labels}

    def inverse_transform(self, y, class_weights=False):
        # Input in different form if coming from target encoder versus preds
        if class_weights:
            # Unpack 0-dim numpy array
            y = y.item()
            # Unpack target encoder output
            groups = y["groups"]
            tags = y["tags"]
        else:
            # Unpack predict module output
            groups, tags = y[0], y[1]

        _classes_ = self.classes_

        self.classes_, self.lookup = self.ner_classes_, self.ner_lookup
        tags = SequenceLabelingEncoder.inverse_transform(self, tags)

        self.classes_ = _classes_

        return (tags, groups)

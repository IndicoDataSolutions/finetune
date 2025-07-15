from abc import ABCMeta
import logging

import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

LOGGER = logging.getLogger("finetune")

class BaseEncoder(metaclass=ABCMeta):
    @property
    def target_labels(self):
        return getattr(self, 'classes_', None)

    @property
    def target_dim(self):
        return len(self.target_labels) if self.target_labels is not None else None



class OneHotLabelEncoder(LabelEncoder, BaseEncoder):

    def _make_one_hot(self, labels):
        output = np.zeros([len(labels), len(self.classes_)], dtype=float)
        output[np.arange(len(labels)), labels] = 1
        return output

    def fit_transform(self, y):
        labels = super().fit_transform(y)
        return self._make_one_hot(labels)

    def transform(self, y):
        labels = super().transform(y)
        return self._make_one_hot(labels)

    def inverse_transform(self, one_hot):
        ys = []
        one_hot = np.asarray(one_hot)
        for row in one_hot:
            for i, flag in enumerate(row):
                if flag == 1:
                    ys.append(self.target_labels[i])
                    break
        return ys


class SequenceLabelingEncoder(BaseEncoder):
    def __init__(self, pad_token, bio_tagging=False, group_tagging=False):
        self.classes_ = None
        self.pad_token = pad_token
        self.lookup = None
        self.bio_tagging = bio_tagging
        self.group_tagging = group_tagging

    def fit(self, labels):
        self.classes_ = sorted(list(set(lab_i["label"] for lab in labels for lab_i in lab) | {self.pad_token}))
        if self.bio_tagging:
            # <PAD> is duplicated here, removed in the set() call
            self.classes_ = [pre + c if c != self.pad_token else c
                             for c in self.classes_ for pre in ("B-", "I-")]
            self.classes_ = sorted(list(set(self.classes_)))
        self.lookup = {c: i for i, c in enumerate(self.classes_)}

    def pre_process_label(self, out, labels):
        pad_idx = self.lookup[self.pad_token]
        return labels, pad_idx

    @staticmethod
    def overlaps(label, tok_start, tok_end, tok_text, input_text, offset=None):
        does_overlap = (
            label["start"] < tok_end <= label["end"] or
            tok_start < label["end"] <= tok_end
        )
        if not does_overlap:
            return False, False

        # Don't run check if text wasn't provided
        if 'text' in label:
            if offset is not None:
                # offsets are present when we are using document labeler.
                # In this case input_text is a page and offset is the char index of where that page starts.
                label_text = label["text"][max(0, offset - label["start"]): len(input_text) + offset - label["start"]]
                doc_text = input_text[max(0, label["start"] - offset): label["end"] - offset]
                strings_agree = doc_text == label_text
            else:
                strings_agree = input_text[label["start"]: label["end"]] == label["text"]
        else:
            strings_agree = True

        return does_overlap, strings_agree

    def transform(self, out, labels):
        # This is basically input_text[0] in the normal case and joins pages for doclabeler
        input_text = "".join(out.input_text)
        labels, pad_idx = self.pre_process_label(out, labels)
        labels_out = [pad_idx for _ in out.tokens]
        offset = out.offset or 0
        bio_pre, group_pre = None, None

        for label in labels:
            current_tag = label["label"]
            current_label = current_tag

            if self.bio_tagging or self.group_tagging:
                bio_pre, group_pre = "", ""
                if self.bio_tagging:
                    bio_pre = "B-"
                if self.group_tagging:
                    if label["group_start"]:
                        group_pre = "BG-"
                    elif label["group_start"] is not None:
                        group_pre = "IG-"
                current_label = f"{group_pre}{bio_pre}{current_tag}"

            for i, (start, end, text) in enumerate(zip(out.token_starts, out.token_ends, out.tokens)):
                # Label extends less than halfway through token
                if label["end"] < (start + end + 1) // 2:
                    break
                overlap, agree = self.overlaps(label, start, end, text, input_text, offset=offset)
                if overlap:
                    if not agree:
                        raise ValueError(
                            "Tokens and labels do not align. {} matches with {}".format(
                                label,
                                input_text[label["start"] - offset: label["end"] - offset]
                            )
                        )
                    if labels_out[i] != pad_idx and self.lookup[current_label] != labels_out[i]:
                        LOGGER.warning("Overlapping labels were found, consider multilabel_sequence=True")
                    if current_label not in self.lookup:
                        LOGGER.warning(
                            "Attempting to encode unknown labels : {}, ignoring for now but this will likely not "
                            "result in desirable behaviour. Available labels are {}".format(current_label, self.lookup.keys())
                        )
                    else:
                        labels_out[i] = self.lookup[current_label]
                        if self.bio_tagging or self.group_tagging:
                            if self.bio_tagging and bio_pre == "B-":
                                bio_pre = "I-"
                            if self.group_tagging and group_pre == "BG-":
                                group_pre = "IG-"
                            current_label = f"{group_pre}{bio_pre}{current_tag}"
        return labels_out

    def inverse_transform(self, y):
        # TODO: update when finetune_to_indico is removed
        return [self.classes_[l] for l in y]

class MultilabelClassificationEncoder(MultiLabelBinarizer, BaseEncoder):
    pass


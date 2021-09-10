from abc import ABCMeta
import logging
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OrdinalEncoder

LOGGER = logging.getLogger("finetune")

class BaseEncoder(metaclass=ABCMeta):
    @property
    def target_labels(self):
        return getattr(self, 'classes_', None)

    @property
    def target_dim(self):
        return len(self.target_labels) if self.target_labels is not None else None


class RegressionEncoder(BaseEncoder):
    def __init__(self):
        self.num_outputs = None

    def fit(self, x):
        self.fit_transform(x)
        return self

    def transform(self, x):
        output = np.array(x)
        rank = len(output.shape)
        if rank == 1:
            return np.expand_dims(output, 1)  # for single output value regression.
        if rank == 2:
            return output
        raise ValueError("Unresolvable shape: {}. Must be able to fit a format [batch, n_outputs]".format(output.shape))

    def fit_transform(self, x):
        output = self.transform(x)
        self.num_outputs = output.shape[1]
        return output

    def inverse_transform(self, y):
        y = np.array(y)
        if y.shape[1] == 1:
            return np.squeeze(y, 1)
        else:
            return y

    @property
    def target_dim(self):
        return self.num_outputs

    @property
    def target_labels(self):
        raise ValueError


class OneHotLabelEncoder(LabelEncoder, BaseEncoder):

    def _make_one_hot(self, labels):
        output = np.zeros([len(labels), len(self.classes_)], dtype=np.float)
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


class NoisyLabelEncoder(LabelEncoder, BaseEncoder):

    # Overriding the fit method...
    # Fit method may not be necessary at all if pandas is
    # consistent about how it chooses columns
    # TODO: Check
    def fit(self, y):
        self.classes_ = list(pd.DataFrame(y[:1]).columns)
        return self

    def transform(self, y):
        return pd.DataFrame(y, columns=self.classes_, dtype=np.float).values

    #TODO: Make output dataframe consistent with self.target_labels
    # and self.classes_
    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, probabilities):
        dataframe = pd.DataFrame(probabilities, columns=self.classes_)
        return list(dataframe.T.to_dict().values())

class Seq2SeqLabelEncoder(BaseEncoder):
    def __init__(self, input_pipeline, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_pipeline = input_pipeline
        self.max_len = max_len

    def fit(self, y):
        return

    @property
    def encoder(self):
        return self.input_pipeline.text_encoder

    @property
    def target_dim(self):
        return self.encoder.vocab_size

    def fit_transform(self, y):
        return self.transform(y)

    def transform(self, y):
        output = []
        for y_i in y:
            out = self.encoder.encode_multi_input([y_i], max_length=self.max_len, include_bos_eos=True).token_ids
            output.append((out, len(out)))
        return output

    def inverse_transform(self, y):
        return [self.encoder.decode(y_i.tolist()) for y_i in y]

class OrdinalRegressionEncoder(OrdinalEncoder, BaseEncoder):

    def __init__(self):
        self.num_outputs = None
        super().__init__()

    def _force_2d(self, x):
        return np.array(x, dtype=np.int32).reshape(-1, 1)

    def fit(self, x):
        super().fit(self._force_2d(x))
        self.num_outputs = len(self.categories_[0]) - 1
        return self

    def transform(self, x):
        labels = super().transform(self._force_2d(x)).astype(np.int32)
        labels = self.rank_to_one_hot(labels)
        return labels

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def rank_to_one_hot(self, x):
        # changes a one-variable rank into an array of 1s and 0s defining the target output of each threshold
        one_hot = np.zeros((len(x), self.num_outputs), dtype=np.float32)
        for i, (rank,) in enumerate(x):
            one_hot[i, :rank] = 1
        return one_hot

    def inverse_transform(self, y):
        y = np.array(y)
        y = y > 0.5
        rank = np.sum(y, axis=1)
        rank = np.expand_dims(rank, 1)
        y = super().inverse_transform(rank)
        y = np.squeeze(y)
        return y

    @property
    def target_dim(self):
        return self.num_outputs

    @property
    def target_labels(self):
        raise ValueError

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
                strings_agree = input_text[label["start"] - offset: label["end"] - offset] == label["text"]
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

class SequenceMultiLabelingEncoder(SequenceLabelingEncoder):
    def transform(self, out, labels):
        labels, _ = self.pre_process_label(out, labels)
        labels_out = [[0 for _ in self.classes_] for _ in out.tokens]
        for i, (start, end) in enumerate(zip(out.token_starts, out.token_ends)):
            for label in labels:
                if label["start"] <= start < label["end"] or label["start"] < end <= label["end"]:
                    if label["label"] not in self.lookup:
                        LOGGER.warning(
                            "Attempting to encode unknown labels, ignoring for now but this will likely not "
                            "result in desirable behaviour"
                        )
                    else:
                        labels_out[i][self.lookup[label["label"]]] = 1
        return labels_out

    def inverse_transform(self, y):
        # TODO: update when finetune_to_indico is removed
        return [tuple(c for c, l_i in zip(self.classes_, l) if l_i) for l in y]


class MultilabelClassificationEncoder(MultiLabelBinarizer, BaseEncoder):
    pass


class IDEncoder(BaseEncoder):

    def __init__(self):
        self.classes_ = [0]

    def transform(self, x):
        return x

    def fit(self, x):
        return x

    def fit_transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

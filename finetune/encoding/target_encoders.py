from abc import ABCMeta

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OrdinalEncoder


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
        fit(labels)
        return transform(labels)

    def inverse_transform(self, probabilities):
        dataframe = pd.DataFrame(probabilities, columns=self.classes_)
        return list(dataframe.T.to_dict().values())

class Seq2SeqLabelEncoder(BaseEncoder):
    def __init__(self, encoder, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.max_len = max_len

    def fit(self, y):
        return

    @property
    def target_dim(self):
        return self.encoder.vocab_size

    def fit_transform(self, y):
        return self.transform(y)

    def transform(self, y):
        output = []
        for y_i in y:
            out = self.encoder.encode_multi_input([[y_i]], max_length=self.max_len).token_ids
            seq_length = len(out)
            x = np.zeros((self.max_len, 2), dtype=np.int32)

            x[:seq_length, 0] = out
            x[:, 1] = np.arange(self.encoder.vocab_size, self.encoder.vocab_size + self.max_len)
            output.append(x)
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


class SequenceLabelingEncoder(LabelEncoder, BaseEncoder):
    pass


class SequenceMultiLabelingEncoder(MultiLabelBinarizer, BaseEncoder):
    pass


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

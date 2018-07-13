import numpy as np
from sklearn.preprocessing import LabelEncoder


class OrdinalClassificationEncoder:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.lookup = None
        self.inverse_lookup = None
        self.keys = None
        self.classes_ = [0, 1]

    def fit(self, y):
        self.keys = list(set(y))
        self.keys.sort()
        spaced_probs = np.linspace(self.min_val, self.max_val, len(self.keys))
        prob_distributions = np.transpose([spaced_probs, 1 - spaced_probs])
        self.inverse_lookup = spaced_probs
        self.lookup = dict(zip(self.keys, prob_distributions))
        return self

    def transform(self, y):
        return list(map(self.lookup.get, y))

    def inverse_transform(self, y):
        output = []
        for item in y:
            i_min = np.argmin(np.abs(self.inverse_lookup - item[0]))
            output.append(self.keys[i_min])
        return np.asarray(output)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    @property
    def target_dim(self):
        return self.classes_


class RegressionEncoder:
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
        self.num_outputs = np.arange(output.shape[1])
        return output

    def inverse_transform(self, y):
        if y.shape[1] == 1:
            return np.squeeze(y, 1)
        else:
            return y

    @property
    def target_dim(self):
        return self.num_outputs


class OneHotLabelEncoder(LabelEncoder):

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

    @property
    def target_dim(self):
        return self.classes_


class SequenceLabelingEncoder(LabelEncoder):

    def fit_transform(self, y):
        shape = np.shape(y)
        flat = np.reshape(y, [-1])
        labels = super().fit_transform(flat)
        return np.reshape(labels, shape)

    def transform(self, y):
        shape = np.shape(y)
        flat = np.reshape(y, [-1])
        labels = super().transform(flat)
        return np.reshape(labels, shape)

    def inverse_transform(self, y):
        shape = np.shape(y)
        flat = np.reshape(y, [-1])
        labels = super().inverse_transform(list(flat))  # list to fix sklearn out of sync with numpy issue
        return np.reshape(labels, shape)

    @property
    def target_dim(self):
        return self.classes_

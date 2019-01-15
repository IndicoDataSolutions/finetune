import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OrdinalEncoder
from abc import ABCMeta

from finetune.utils import flatten


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

class OrdinalRegressionEncoder(OrdinalEncoder, BaseEncoder):
    
    def __init__(self):
        self.num_outputs = None
        super().__init__()
        
    def fit(self, x):
        x = np.array(x)
        rank = len(x.shape)
        if rank == 1:
            x = np.expand_dims(x, 1)
        self.fit_transform(x)
        return self

    def transform(self, x):
        x = np.array(x)
        rank = len(x.shape)
        if rank == 1:
            x = np.expand_dims(x, 1)  # for single output value regression.
        labels = super().transform(x)
        labels = self.rank_to_thresholds(labels)
        return labels
        
    def fit_transform(self, x):
        super().fit(x)
        labels = self.transform(x)
        self.num_outputs = labels.shape[1]
        return labels
    
    def rank_to_thresholds(self,x):
        #changes a one-variable rank into an array of 1s and 0s defining the target output of each threshold
        x = x[:10000]
        num_thresholds = len(self.categories_[0])-1
        thresholds = [np.concatenate((np.ones(int(rank)),np.zeros((num_thresholds-int(rank))))) for rank in x]
        return np.array(thresholds)

    def inverse_transform(self, y):
        #this commented part doesn't work yet
        #y = np.asarray(y)
        #y = y > 0.5
        #y = super().inverse_transform(y)
        y = np.sum(y,axis = 1)
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

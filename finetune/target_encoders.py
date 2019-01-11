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
        
    def get_thresholds(self, y):
        """
        Creates list of rankings. If all distances between rankings are the same,
        output is unit ranking (e.g. 1, 2, 3...) . In any other case, the smallest 
        distance between rankings is set to unity and other distances scaled by an 
        appropriate factor.
        
        classes = list(set(y))
        classes.sort()
        num_classes = len(classes)
        scaled = np.ones(num_classes)
        diffs = [classes[i+1] - classes[i] for i in range(num_classes-1)]
        print(diffs)
        diffs = [0] + [diffs[i]/min(diffs) for i in range(num_classes-1)]
        print(diffs)
        for i in range(1,num_classes):
            scaled[i] +=  np.sum(diffs[:i+1])
        return scaled
        """
        
        #for now, assume normal rankings (positive, constant distance between each other)
        classes = list(set(y))
        classes.sort()
        return classes
        
    def fit(self, x):
        x = np.array(x)
        rank = len(x.shape)
        if rank == 1:
            x = np.expand_dims(x, 1)
        super().fit(x)
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
        labels = super().fit_transform(x)
        labels = self.rank_to_thresholds(labels)
        self.num_outputs = labels.shape[1]
        return labels
    
    def rank_to_thresholds(self,x):
        #changes a one-variable rank into an array of 1s and 0s defining the target output of each threshold
        num_thresholds = len(self.categories_[0])-1
        thresholds = []
        #print("NUM: {}".format(num_thresholds))
        #print("RANK: {}".format(x[0]))
        for i in range(len(x)):
            rank = int(x[i])
            thresholds.append(np.concatenate((np.ones(rank,),np.zeros((num_thresholds-rank),))))
        return thresholds

    def inverse_transform(self, y):
        y = super().inverse_transform(y)
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

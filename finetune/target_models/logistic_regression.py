import spacy
import numpy as np
import joblib as jl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from finetune.base import BaseModel
from finetune.errors import FinetuneError

class GloveEncoder():
    nlp = spacy.load("en_core_web_lg")

    def encode(self, text, train):
        batch_glove = np.squeeze(np.asarray([self.nlp(example).vector for example in text]))
        return batch_glove

class TfidfEncoder():

    def __init__(self):
        self.encoder = TfidfVectorizer(ngram_range=(1,3), max_features=5000)

    def encode(self, text, train):
        if train: self.encoder.fit(text)
        batch_tfidf = np.squeeze(np.asarray(self.encoder.transform(text).toarray()))
        return batch_tfidf


class LogisticRegressionClassifier():
    """ 
    Classifies a single document into 1 of N categories using logistic regression.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """
    defaults = {'encoder': 'glove'}
    

    def __init__(self, **kwargs):
        self.LR = LogisticRegression(penalty="l2", max_iter=500, solver="lbfgs", multi_class='multinomial')
        self.LR.C = kwargs.pop('C', None) or self.LR.class_weight
        self.LR.class_weight = kwargs.pop('class_weights', None) or self.LR.class_weight

        encoder = kwargs.pop('encoder', self.defaults['encoder'])
        if encoder not in ['glove', 'tfidf']:
            raise FinetuneError("Invalid encoder setting {} given: Must be in {'glove', 'tfidf'}".format(encoder))
        self.encoder = GloveEncoder() if encoder == 'glove' else TfidfEncoder()

        self.been_fit = False

    def featurize(self, X):
        X = self.encoder.encode(X, train=False)
        if len(np.shape(X)) == 1: X = np.expand_dims(X, axis=0)
        return X

    def predict(self, X):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        if not self.been_fit: raise FinetuneError("Cannot predict until model has been fit.")
        X = self.encoder.encode(X, train=False)
        if len(np.shape(X)) == 1: X = np.expand_dims(X, axis=0)
        return self.LR.predict(X)

    def predict_proba(self, X):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        if not self.been_fit: raise FinetuneError("Cannot predict_proba until model has been fit.")
        X = self.encoder.encode(X, train=False)
        if len(np.shape(X)) == 1: X = np.expand_dims(X, axis=0)
        preds = self.LR.predict_proba(X)
        dict_preds = [{label:prob for label, prob in zip(self.LR.classes_, pred)} for pred in preds]
        return dict_preds

    def finetune(self, X, Y=None, batch_size=None):
        """
        :param X: list or array of text.
        :param Y: integer or string-valued class labels.
        """
        return self.fit(X, Y=Y)

    def fit(self, X, Y=None, batch_size=None):
        if len(X) != len(Y):
            raise FinetuneError("X and Y have different lengths.")
        X = self.encoder.encode(X, train=True)
        self.LR.fit(X, Y)
        self.been_fit = True

    @classmethod
    def load(self, path):
        finetune_obj = jl.load(path)
        return finetune_obj

    def save(self, path):
        jl.dump(self, path)

    def explain(self, Xs):
        raise AttributeError("`LogisticRegression` model does not support `explain`.")

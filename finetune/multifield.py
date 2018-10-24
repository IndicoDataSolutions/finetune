from finetune.classifier import Classifier, ClassificationPipeline
from finetune.regressor import Regressor, RegressionPipeline
from finetune.base import BaseModel

class MultifieldClassificationPipeline(ClassificationPipeline):
    def _format_for_encoding(self, X):
        return [X]


class MultifieldRegressionPipeline(RegressionPipeline):
    def _format_for_encoding(self, X):
        return [X]


class MultifieldClassifier(Classifier):
    """ 
    Classifies a set of documents into 1 of N classes.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def _get_input_pipeline(self):
        return MultifieldClassificationPipeline(self.config)
        
    def finetune(self, Xs, Y=None, batch_size=None):
        """
        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return BaseModel.finetune(self, Xs, Y=Y, batch_size=batch_size)

    def predict(self, Xs):
        """
        Produces list of most likely class labels as determined by the fine-tuned model.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of class labels.
        """
        return BaseModel.predict(self, Xs)

    def predict_proba(self, Xs):
        """
        Produces probability distribution over classes for each example in X.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of dictionaries.  Each dictionary maps from X2 class label to its assigned class probability.
        """
        return BaseModel.predict_proba(self, Xs)

    def featurize(self, Xs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, Xs)


class MultifieldRegressor(Regressor):
    """ 
    Regresses one or more floating point values given a set of documents per example.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def _get_input_pipeline(self):
        return MultifieldRegressionPipeline(self.config)
        
    def finetune(self, Xs, Y=None, batch_size=None):
        """
        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :param Y: floating point targets
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return BaseModel.finetune(self, Xs, Y=Y, batch_size=batch_size)

    def predict(self, Xs):
        """
        Produces list of most likely class labels as determined by the fine-tuned model.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of class labels.
        """
        return BaseModel.predict(self, Xs)

    def predict_proba(self, Xs):
        """
        Produces probability distribution over classes for each example in X.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of dictionaries.  Each dictionary maps from X2 class label to its assigned class probability.
        """
        return BaseModel.predict_proba(self, Xs)

    def featurize(self, Xs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, Xs)
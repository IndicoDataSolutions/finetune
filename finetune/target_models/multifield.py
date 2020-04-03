import copy

from finetune.errors import FinetuneError
from finetune.target_models.classifier import Classifier, ClassificationPipeline
from finetune.target_models.regressor import Regressor, RegressionPipeline
from finetune.base import BaseModel


class MultiFieldClassificationPipeline(ClassificationPipeline):
    def _format_for_encoding(self, X):
        return X


class MultiFieldRegressionPipeline(RegressionPipeline):
    def _format_for_encoding(self, X):
        return X


class MultiFieldClassifier(Classifier):
    """ 
    Classifies a set of documents into 1 of N classes.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    defaults = {"chunk_long_sequences": False}

    def __init__(self, **kwargs):
        d = copy.deepcopy(MultiFieldClassifier.defaults)
        d.update(kwargs)
        super().__init__(**d)
        if self.config.chunk_long_sequences:
            raise FinetuneError(
                "Multifield model is incompatible with chunk_long_sequences = True in config."
            )

    def _get_input_pipeline(self):
        return MultiFieldClassificationPipeline(self.config)

    def finetune(self, Xs, Y=None, context=None, **kwargs):
        """
        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return BaseModel.finetune(self, Xs, Y=Y, context=context, **kwargs)

    def predict(self, Xs, context=None, **kwargs):
        """
        Produces list of most likely class labels as determined by the fine-tuned model.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of class labels.
        """
        return BaseModel.predict(self, Xs, context=context, **kwargs)

    def predict_proba(self, Xs, context=None, **kwargs):
        """
        Produces probability distribution over classes for each example in X.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of dictionaries.  Each dictionary maps from X2 class label to its assigned class probability.
        """
        return BaseModel.predict_proba(self, Xs, context=context, **kwargs)

    def featurize(self, Xs, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, Xs, **kwargs)


class MultiFieldRegressor(Regressor):
    """ 
    Regresses one or more floating point values given a set of documents per example.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def _get_input_pipeline(self):
        return MultiFieldRegressionPipeline(self.config)

    def finetune(self, Xs, Y=None, **kwargs):
        """
        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :param Y: floating point targets
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return BaseModel.finetune(self, Xs, Y=Y, **kwargs)

    def predict(self, Xs, **kwargs):
        """
        Produces list of most likely class labels as determined by the fine-tuned model.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of class labels.
        """
        return BaseModel.predict(self, Xs, **kwargs)

    def predict_proba(self, Xs, **kwargs):
        """
        Produces probability distribution over classes for each example in X.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: list of dictionaries.  Each dictionary maps from X2 class label to its assigned class probability.
        """
        return BaseModel.predict_proba(self, Xs, **kwargs)

    def featurize(self, Xs, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param \*Xs: lists of text inputs, shape [batch, n_fields]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, Xs, **kwargs)

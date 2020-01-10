from finetune.base import PredictMode
from finetune.target_models.classifier import Classifier


class LanguageModel(Classifier):
    """ 
    A Language Model for Finetune

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def predict(self, X, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: Perplexities of each of the input sentences.
        """
        return self._inference(X, predict_keys=[PredictMode.LM_PERPLEXITY], context=context, **kwargs)

    def predict_proba(self, X, context=None):
        raise ValueError("Predict Proba is not defined for the language model")

    def finetune(self, X, Y=None, batch_size=None, context=None, **kwargs):
        """
        :param X: list or array of text.
        :param Y: Not used.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        if Y:
            raise ValueError("No targets should be provided for the language model")
        return super().finetune(X, Y=None, batch_size=batch_size, **kwargs)

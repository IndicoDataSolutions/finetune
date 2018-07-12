from finetune.config import BATCH_SIZE
from finetune.lm_base import LanguageModelBase


class LanguageModelSequenceLabeling(LanguageModelBase):

    def featurize(self, X, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X, max_length=max_length)

    def predict(self, X, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return self._predict(X, max_length=max_length)

    def predict_proba(self, X, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return self._predict_proba(X, max_length=max_length)

    def finetune(self, X, Y, batch_size=BATCH_SIZE, val_size=0.05, val_interval=150):
        """
        :param X: list or array of text.
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        :param val_size: Float fraction or int number that represents the size of the validation set.
        :param val_interval: The interval for which validation is performed, measured in number of steps.
        """
        self.is_classification = True
        return self._finetune(X, Y=Y, batch_size=batch_size, val_size=val_size, val_interval=val_interval)
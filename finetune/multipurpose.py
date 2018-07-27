import numpy as np

from finetune.base import BaseModel, CLASSIFICATION, REGRESSION, SEQUENCE_LABELING
from finetune.errors import InvalidTargetType


class Model(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _text_to_ids(self, *Xs, max_length=None):
        max_length = max_length or self.config.max_length
        Xs = [[[x] for x in X] for X in Xs]
        encoded_output = self.encoder.encode_multi_input(*Xs, max_length=max_length, verbose=self.config.verbose)
        return self._array_format(encoded_output)

    def finetune(self, Xs, Y, batch_size=None):
        """
        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        if self.target_type is None:
            if np.array(Y).dtype == 'float':
                self.target_type = REGRESSION
            elif len(Y.shape) == 1:  # [batch]
                self.target_type = CLASSIFICATION
            else:
                raise InvalidTargetType(
                    "targets must either be a 1-d array of classification targets or a "
                    "2-d array of sequence labels."
                )

        return self._finetune(*list(zip(*Xs)), Y=Y, batch_size=batch_size)

    def predict(self, Xs, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return self._predict(*list(zip(*Xs)), max_length=max_length)

    def predict_proba(self, Xs, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return self._predict_proba(*list(zip(*Xs)), max_length=max_length)

    def featurize(self, Xs, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(*list(zip(*Xs)), max_length=max_length)

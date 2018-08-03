import numpy as np

from finetune.base import BaseModel
from finetune.encoding import ArrayEncodedOutput
from finetune.target_encoders import IDEncoder
import tensorflow as tf

from finetune.network_modules import multi_choice_question


class QandA(BaseModel):
    """
    Multi choice question finetune model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = None

    def _text_to_ids(self, question, answers, Y=None, max_length=None):
        """
        Format multi question examples as a list of IDs
        """
        arrays = [super(QandA, self)._text_to_ids(question, ans) for ans in answers]
        kwargs = arrays[0]._asdict()
        kwargs['tokens'] = [arr.tokens for arr in arrays]
        kwargs['token_ids'] = np.stack([arr.token_ids for arr in arrays], 1)
        kwargs['mask'] = np.stack([arr.mask for arr in arrays], 1)
        return ArrayEncodedOutput(**kwargs)

    def finetune(self, question, correct_answer, incorrect_answers, batch_size=None, fit_lm_only=False):
        """
        :param question: List or array of text, shape [batch]
        :param correct_answer: List or array of correct answers [batch]
        :param incorrect_answers: List or array of text, shape [n_answers - 1 , batch]
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        self.num_answers = len(incorrect_answers) + 1
        arr_encoded = self._text_to_ids(question, [correct_answer] + incorrect_answers)
        labels = None if fit_lm_only else np.zeros(len(question), dtype=np.int)
        return self._training_loop(arr_encoded, Y=labels, batch_size=batch_size)

    def _define_placeholders(self):
        super()._define_placeholders()
        self.X = tf.placeholder(tf.int32, [None, self.num_answers, self.config.max_length, 2])
        self.M = tf.placeholder(tf.float32, [None, self.num_answers, self.config.max_length])  # sequence mask
        self.Y = tf.placeholder(tf.int32, [None])

    def _target_encoder(self):
        return IDEncoder()

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return multi_choice_question(
            hidden=featurizer_state['features'],
            targets=targets,
            n_targets=self.num_answers,
            dropout_placeholder=self.do_dropout,
            config=self.config,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        return tf.argmax(logits, -1)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

    def predict(self, question, answers, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [n_answers, batch]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        raw_ids = BaseModel.predict(self, question, answers, max_length=max_length)
        return [ans[i] for ans, i in zip(zip(*answers), raw_ids)]

    def predict_proba(self, question, answers, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [n_answers, batch]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        raw_probas = self._predict_proba(question, answers, max_length)

        formatted_predictions = []
        for probas, *answers_per_sample in zip(raw_probas, *answers):
            formatted_predictions.append(
                dict(zip(answers_per_sample, probas))
            )
        return formatted_predictions

    def featurize(self, question, answers, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [n_answers, batch]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, question, answers, max_length=max_length)

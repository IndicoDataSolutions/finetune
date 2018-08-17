import numpy as np

from finetune.base import BaseModel
from finetune.encoding import ArrayEncodedOutput
from finetune.target_encoders import IDEncoder
import tensorflow as tf

from finetune.network_modules import multi_choice_question
from finetune.utils import list_transpose


class MultipleChoice(BaseModel):
    """
    Multi choice question finetune model.
    
    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = None

    def _text_to_ids(self, Xs, Y=None, max_length=None):
        """
        Format multi question examples as a list of IDs
        """
        qa_pairs = [
            [q, answer_list[0]] 
            for q, answer_list in Xs
        ]
        arrays = [
            super(MultipleChoice, self)._text_to_ids(
                [
                    [q, answer_list[idx]] 
                    for q, answer_list in Xs
                ],
                Y=Y
            ) 
            for idx in range(self.num_answers)
        ]
        kwargs = arrays[0]._asdict()
        kwargs['tokens'] = [arr.tokens for arr in arrays]
        kwargs['token_ids'] = np.stack([arr.token_ids for arr in arrays], 1)
        kwargs['mask'] = np.stack([arr.mask for arr in arrays], 1)
        return ArrayEncodedOutput(**kwargs)

    def finetune(self, questions, answers, correct_answer, batch_size=None, fit_lm_only=False, max_length=None):
        """
        :param questions: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers], must contain the correct answer for each entry.
        :param correct_answer: List or array of correct answers [batch] either in the format of an idx to the correct
                answer or a string of the correct answer.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        max_length = max_length or self.config.max_length

        answer_idx = []
        if not len(correct_answer) == len(answers) == len(questions):
            raise ValueError("Answers, questions and corrext_answer are not all the same length, {},{},{}".format(
                len(questions), len(correct_answer), len(answers)
            ))
     
        for correct, others in zip(correct_answer, answers):
            if isinstance(correct, int):
                if 0 > correct > len(others):
                    raise ValueError(
                        "Correct answer is of type int but is invalid with value {} for answers of len {}".format(
                            correct, len(others)))
                answer_idx.append(correct)
            else:
                try:
                    ans_idx = others.index(correct)
                    answer_idx.append(ans_idx)
                except ValueError:
                    raise ValueError(
                        "Correct answer {} is not contained in possible answers {}".format(correct, others))

        self.num_answers = len(answers[0])
        arr_encoded = self._text_to_ids(list(zip(questions, answers)), max_length=max_length)
        labels = None if fit_lm_only else answer_idx
        return self._training_loop(arr_encoded, Y=labels, batch_size=batch_size)

    def _define_placeholders(self, *args, **kwargs):
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

    def predict(self, questions, answers, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        raw_ids = BaseModel.predict(self, list(zip(questions, answers)), max_length=max_length)
        return [ans[i] for ans, i in zip(zip(*answers), raw_ids)]

    def predict_proba(self, questions, answers, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        answers = list_transpose(answers)
        raw_probas = self._predict_proba(zip(questions, answers), max_length=max_length)

        formatted_predictions = []
        for probas, *answers_per_sample in zip(raw_probas, *answers):
            formatted_predictions.append(
                dict(zip(answers_per_sample, probas))
            )
        return formatted_predictions

    def featurize(self, questions, answers, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param questions: List or array of text, shape [batch]
        :param answers: List or array of text, shape [n_answers, batch]
        :param max_length: the number of byte-pair encoded tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, zip(questions, answers), max_length=max_length)

import numpy as np

from finetune.base import BaseModel
from finetune.input_pipeline import BasePipeline
from finetune.encoding import ArrayEncodedOutput
from finetune.target_encoders import IDEncoder
import tensorflow as tf

from finetune.network_modules import multi_choice_question
from finetune.utils import list_transpose

class MultipleChoicePipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = None


    def _text_to_ids(self, Xs, Y=None, pad_token=None):
        """
        Format multi question examples as a list of IDs
        """
        q, answer_list = Xs
        pairs = [[q, answer_list[idx]] for idx in range(len(answer_list))]
        arrays = []
        for pair in pairs:
            arrays.append(next(super()._text_to_ids(pair, Y=Y)))

        kwargs = arrays[0]._asdict()
        kwargs['tokens'] = [arr.tokens for arr in arrays]
        kwargs['token_ids'] = np.stack([arr.token_ids for arr in arrays], 0)
        kwargs['mask'] = np.stack([arr.mask for arr in arrays], 0)
        yield ArrayEncodedOutput(**kwargs)

    def _format_for_encoding(self, X):
        return [X]

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        return ({"tokens": tf.int32, "mask": tf.float32}, tf.int32), (
            {"tokens": TS([self.num_answers, self.config.max_length, 2]), "mask": TS([self.num_answers, self.config.max_length])}, TS([]))

    def _target_encoder(self):
        return IDEncoder()


class MultipleChoice(BaseModel):
    """
    Multi choice question finetune model.
    
    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = None

    def _get_input_pipeline(self):
        return MultipleChoicePipeline(self.config)

    def finetune(self, questions, answers, correct_answer, fit_lm_only=False):
        """
        :param questions: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers], must contain the correct answer for each entry.
        :param correct_answer: List or array of correct answers [batch] either in the format of an idx to the correct
                answer or a string of the correct answer.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
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
        self.input_pipeline.num_answers = self.num_answers #TODO(BEN) factor this inside the post_data_init
        labels = None if fit_lm_only else answer_idx
        return super().finetune(list(zip(questions, answers)), Y=labels)

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return multi_choice_question(
            hidden=featurizer_state['features'],
            targets=targets,
            n_targets=self.num_answers,
            config=self.config,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        return tf.argmax(logits, -1)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

    def predict(self, questions, answers):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers]
        :returns: list of class labels.
        """
        raw_ids = BaseModel.predict(self, list(zip(questions, answers)))
        return [ans[i] for ans, i in zip(answers, raw_ids)]

    def predict_proba(self, questions, answers):
        """
        Produces a probability distribution over classes for each example in X.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers]
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        answers = list_transpose(answers)
        raw_probas = self._predict_proba(zip(questions, answers))

        formatted_predictions = []
        for probas, *answers_per_sample in zip(raw_probas, *answers):
            formatted_predictions.append(
                dict(zip(answers_per_sample, probas))
            )
        return formatted_predictions

    def featurize(self, questions, answers):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param questions: List or array of text, shape [batch]
        :param answers: List or array of text, shape [n_answers, batch]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, zip(questions, answers))

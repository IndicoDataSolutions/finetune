import numpy as np

from finetune.base import BaseModel
from finetune.input_pipeline import BasePipeline
from finetune.encoding.input_encoder import EncodedOutput
from finetune.encoding.target_encoders import IDEncoder
import tensorflow as tf

from finetune.nn.target_blocks import multi_choice_question
from finetune.util import list_transpose
from finetune.encoding.input_encoder import tokenize_context
from finetune.model import PredictMode

def padded_stack(arrays):
    lens = [arr.shape[0] for arr in arrays]
    other_padding = [(0, 0) for _ in arrays[0].shape[1:]]
    max_len = max(lens)
    padded = [np.pad(a, ((0, max_len - l), *other_padding), "constant") for a, l in zip(arrays, lens)]
    print([p.shape for p in arrays])
    print([p.shape for p in padded])
    return np.stack(padded, 0)


class MultipleChoicePipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _post_data_initialization(self, Y):
        super()._post_data_initialization(Y)
        self.target_dim = self.target_dim_

    def _text_to_ids(self, Xs, pad_token=None):
        """
        Format multi question examples as a list of IDs
        """
        q, answer_list = Xs

        pairs = [[q, answer_list[idx]] for idx in range(len(answer_list))]
        arrays = []
        for pair in pairs:
            arrays.append(next(super()._text_to_ids(pair)))

        kwargs = arrays[0]._asdict()
        max_len = max([len(arr.token_ids) for arr in arrays])
        kwargs["tokens"] = [arr.tokens for arr in arrays]
        kwargs["token_ids"] = padded_stack([arr.token_ids for arr in arrays])
        yield EncodedOutput(**kwargs)

    def text_to_tokens_mask(self, X, Y=None, context=None):
        out_gen = self._text_to_ids(X, pad_token=self.config.pad_token)
        for i, out in enumerate(out_gen):
            if context is None:
                feats = {"tokens": out.token_ids}
            else:
                num_answers = len(out.tokens)
                tokenized_context = []
                for answer_idx in range(num_answers):
                    out_instance = EncodedOutput(
                        token_ids=out.token_ids[answer_idx],
                        tokens=out.token_ids[answer_idx],
                        token_ends=out.token_ends,
                        token_starts=out.token_starts,
                    )
                    context_instance = context[0] + context[answer_idx + 1]
                    tokenized_context.append(tokenize_context(context_instance, out_instance, self.config))
                feats = {"tokens": out.token_ids, "context": tokenized_context}
            if Y is None:
                yield feats
            else:
                yield feats, self.label_encoder.transform([Y])[0]


    def _format_for_encoding(self, X):
        return X

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {
            "tokens": TS([self.target_dim, None]),
        }
        if self.config.use_auxiliary_info:
            TS = tf.TensorShape
            types["context"] = tf.float32
            shapes["context"] = TS([self.target_dim, None, self.config.context_dim])
        return (
            (types, tf.float32,),
            (shapes, TS([]),),
        )


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

    def finetune(self, questions, answers, correct_answer, fit_lm_only=False, context=None, **kwargs):
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
            raise ValueError(
                "Answers, questions and corrext_answer are not all the same length, {},{},{}".format(
                    len(questions), len(correct_answer), len(answers)
                )
            )

        for correct, others in zip(correct_answer, answers):
            if isinstance(correct, int):
                if 0 > correct > len(others):
                    raise ValueError(
                        "Correct answer is of type int but is invalid with value {} for answers of len {}".format(
                            correct, len(others)
                        )
                    )
                answer_idx.append(correct)
            else:
                try:
                    ans_idx = others.index(correct)
                    answer_idx.append(ans_idx)
                except ValueError:
                    raise ValueError(
                        "Correct answer {} is not contained in possible answers {}".format(
                            correct, others
                        )
                    )

        labels = None if fit_lm_only else answer_idx
        self.input_pipeline.target_dim_ = len(answers[0])
        return super().finetune(list(zip(questions, answers)), Y=labels, context=context, **kwargs)

    def _pre_target_model_hook(self, featurizer_state):
        if "context" in featurizer_state:
            context_embed = featurizer_state["context"]
            featurizer_state['features'] = tf.concat(
                (featurizer_state['features'], tf.reduce_mean(input_tensor=context_embed, axis=2)), -1
            )

    def _target_model(
        self, *, config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs
    ):
        return multi_choice_question(
            hidden=featurizer_state["features"],
            targets=targets,
            n_targets=n_outputs,
            config=config,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        return tf.argmax(input=logits, axis=-1)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

    def predict(self, questions, answers, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers]
        :returns: list of class labels.
        """
        zipped_data = self.input_pipeline.zip_list_to_dict(X=list(zip(questions, answers)), context=context)
        raw_preds = self._inference(zipped_data, predict_keys=[PredictMode.NORMAL], context=context, **kwargs)
        raw_ids = self.input_pipeline.label_encoder.inverse_transform(np.asarray(raw_preds))
        return [ans[i] for ans, i in zip(answers, raw_ids)]

    def predict_proba(self, questions, answers, context=None, **kwargs):
        """
        Produces a probability distribution over classes for each example in X.


        :param question: List or array of text, shape [batch]
        :param answers: List or array of text, shape [batch, n_answers]
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        answers = list_transpose(answers)
        zipped_data = self.input_pipeline.zip_list_to_dict(X=list(zip(questions, answers)), context=context)
        raw_probas = self._inference(zipped_data, predict_keys=[PredictMode.PROBAS], **kwargs)
        formatted_predictions = []
        for probas, *answers_per_sample in zip(raw_probas, *answers):
            formatted_predictions.append(dict(zip(answers_per_sample, probas)))
        return formatted_predictions

    def featurize(self, questions, answers, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param questions: List or array of text, shape [batch]
        :param answers: List or array of text, shape [n_answers, batch]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(self, zip(questions, answers), **kwargs)

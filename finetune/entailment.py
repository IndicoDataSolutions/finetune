import json

import tensorflow as tf
from sklearn.model_selection import train_test_split

from finetune.base import BaseModel
from finetune.target_encoders import OneHotLabelEncoder
from finetune.network_modules import classifier


class Entailment(BaseModel):

    def finetune(self, X1, X2, Y=None, batch_size=None):
        """
        :param X1: list or array of text to embed as the queries.
        :param X2: list or array of text to embed as the answers.
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X1, X2, Y=Y, batch_size=batch_size)

    def predict(self, X1, X2, max_length=None):
        """
        Produces X2 list of most likely class labels as determined by the fine-tuned model.

        :param X1: list or array of text to embed as the queries.
        :param X2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return super().predict(X1, X2, max_length=max_length)

    def predict_proba(self, X1, X2, max_length=None):
        """
        Produces X2 probability distribution over classes for each example in X.

        :param X1: list or array of text to embed as the queries.
        :param X2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from X2 class label to its assigned class probability.
        """
        return super().predict_proba(X1, X2, max_length=max_length)

    def featurize(self, X1, X2, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X1: list or array of text to embed as the queries.
        :param X2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X1, X2, max_length=max_length)

    def _target_encoder(self):
        return OneHotLabelEncoder()

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return classifier(
            hidden=featurizer_state['features'], 
            targets=targets, 
            n_targets=n_outputs, 
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


if __name__ == "__main__":

    with open("data/questions.json", "rt") as fp:
        data = json.load(fp)

    scores = []
    questions = []
    answers = []
    save_path = 'saved-models/cola'

    model = Entailment()

    for item in data:
        row = data[item]
        scores.append(row["score"])
        questions.append(row["question"])
        answers.append(row["answers"][0]["answer"])

    scores_train, scores_test, ques_train, ques_test, ans_train, ans_test = train_test_split(
        scores, questions, answers, test_size=0.33, random_state=5)

    model.finetune(ques_train, ans_train, scores_train)

    model = Entailment.load(save_path)

    print("TRAIN EVAL")
    predictions = model.predict(ques_train, ans_train)
    print(predictions)

    from scipy.stats import spearmanr

    print(spearmanr(predictions, scores_train))

    print("TEST EVAL")
    predictions = model.predict(ques_test, ans_test)
    print(predictions)
    print(spearmanr(predictions, scores_test))

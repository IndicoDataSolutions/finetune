import json

from sklearn.model_selection import train_test_split

from finetune.lm_base import LanguageModelBase
from finetune.target_encoders import OrdinalClassificationEncoder


class LanguageModelEntailment(LanguageModelBase):

    def get_target_encoder(self):
        return OrdinalClassificationEncoder()

    def _text_to_ids(self, *Xs, max_length=None):
        max_length = max_length or self.hparams.max_length
        assert len(Xs) == 2, "This implementation assumes 2 Xs"

        question_answer_pairs = self.encoder.encode_for_entailment(*Xs, max_length=max_length, verbose=self.verbose)

        tokens, mask = self._array_format(question_answer_pairs)
        return tokens, mask

    def finetune(self, X_1, X_2, Y=None, batch_size=None):
        """
        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        self.is_classification = True
        return self._finetune(X_1, X_2, Y=Y, batch_size=batch_size)

    def predict(self, X_1, X_2, max_length=None):
        """
        Produces X_2 list of most likely class labels as determined by the fine-tuned model.

        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return self.label_encoder.inverse_transform(self._predict_proba(X_1, X_2, max_length=max_length))

    def predict_proba(self, X_1, X_2, max_length=None):
        """
        Produces X_2 probability distribution over classes for each example in X.

        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from X_2 class label to its assigned class probability.
        """
        return self._predict_proba(X_1, X_2, max_length=max_length)

    def featurize(self, X_1, X_2, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X_1, X_2, max_length=max_length)


if __name__ == "__main__":

    with open("data/questions.json", "rt") as fp:
        data = json.load(fp)

    scores = []
    questions = []
    answers = []
    save_path = 'saved-models/cola'
    model = LanguageModelEntailment(save_path)

    for item in data:
        row = data[item]
        scores.append(row["score"])
        questions.append(row["question"])
        answers.append(row["answers"][0]["answer"])

    scores_train, scores_test, ques_train, ques_test, ans_train, ans_test = train_test_split(
        scores, questions, answers, test_size=0.33, random_state=5)

    #model.finetune(ques_train, ans_train, scores_train)

    model = LanguageModelEntailment.load(save_path)

    print("TRAIN EVAL")
    predictions = model.predict(ques_train, ans_train)
    print(predictions)

    from scipy.stats import spearmanr

    print(spearmanr(predictions, scores_train))

    print("TEST EVAL")
    predictions = model.predict(ques_test, ans_test)
    print(predictions)
    print(spearmanr(predictions, scores_test))

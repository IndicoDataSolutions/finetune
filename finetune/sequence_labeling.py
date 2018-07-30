import tensorflow as tf
from sklearn.model_selection import train_test_split

from finetune.base import BaseModel
from finetune.encoding import EncodedOutput, ArrayEncodedOutput
from finetune.target_encoders import SequenceLabelingEncoder
from finetune.network_modules import sequence_labeler
from finetune.utils import indico_to_finetune_sequence, finetune_to_indico_sequence, sequence_decode


class SequenceLabeler(BaseModel):

    def finetune(self, X, Y=None, batch_size=None):
        """
        :param X: A list of text snippets. Format: [batch_size]
        :param Y: A list of lists of annotations. Format: [batch_size, n_annotations], where each annotation is of the form:
            {'start': 0, 'end': 5, 'label': 'class', 'text': 'sample text'}
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        :param val_size: Float fraction or int number that represents the size of the validation set.
        :param val_interval: The interval for which validation is performed, measured in number of steps.
        """
        fit_language_model_only = (Y is None)
        X, Y = indico_to_finetune_sequence(X, Y, none_value="<PAD>")
        arr_encoded = self._text_to_ids(X, Y=Y)
        labels = None if fit_language_model_only else arr_encoded.labels
        return self._training_loop(arr_encoded, Y=labels, batch_size=batch_size)

    def predict(self, X, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncatindiion.
        :returns: list of class labels.
        """
        doc_subseqs, _ = indico_to_finetune_sequence(X)
        arr_encoded = self._text_to_ids(doc_subseqs)
        labels = self._predict(doc_subseqs, max_length=max_length)
        all_subseqs = []
        all_labels = []
        for text, label_seq, position_seq in zip(X, labels, arr_encoded.char_locs):
            start_of_token = 0
            doc_subseqs = []
            doc_labels = []

            for label, position in zip(label_seq, position_seq):
                if position == -1:
                    # indicates padding / special tokens
                    continue

                # if there are no current subsequence
                # or the current subsequence has the wrong label
                if not doc_subseqs or label != doc_labels[-1]:
                    # start new subsequence
                    doc_subseqs.append(text[start_of_token:position])
                    doc_labels.append(label)
                else:
                    # continue appending to current subsequence
                    doc_subseqs[-1] += text[start_of_token:position]

                start_of_token = position
            all_subseqs.append(doc_subseqs)
            all_labels.append(doc_labels)
        doc_texts, doc_annotations = finetune_to_indico_sequence(raw_texts=X, subseqs=all_subseqs, labels=all_labels)
        return doc_annotations

    def featurize(self, X, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X, max_length=max_length)

    def predict_proba(self, X, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncatindiion.
        :returns: list of class labels.
        """
        doc_subseqs, _ = indico_to_finetune_sequence(X)
        arr_encoded = self._text_to_ids(doc_subseqs)
        batch_probas = self._predict_proba(doc_subseqs, max_length=max_length)
        result = []
        for token_seq, proba_seq in zip(arr_encoded.tokens, batch_probas):
            seq_result = []
            for token, proba_t in zip(token_seq, proba_seq):
                seq_result.append((
                    token,
                    dict(zip(self.label_encoder.classes_, proba_t))
                ))
            result.append(seq_result)
        return result

    def _format_for_encoding(self, *Xs):
        """
        No op -- the default input format is the same format used by SequenceLabeler
        """
        return Xs

    def _target_placeholder(self):
        return tf.placeholder(tf.int32, [None, self.config.max_length])  # classification targets

    def _target_encoder(self):
        return SequenceLabelingEncoder()

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return sequence_labeler(
            hidden=featurizer_state['sequence_features'],
            targets=targets, 
            n_targets=n_outputs,
            dropout_placeholder=self.do_dropout,
            config=self.config,
            train=train, 
            reuse=reuse, 
            **kwargs
        )
    
    def _predict_op(self, logits, **kwargs):
        label_idxs, _ = sequence_decode(logits, kwargs.get("transition_matrix"))
        return label_idxs

    def _predict_proba_op(self, logits, **kwargs):
        _, label_probas = sequence_decode(logits, kwargs.get("transition_matrix"))
        return label_probas

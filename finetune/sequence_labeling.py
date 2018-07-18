import numpy as np

from finetune.base import BaseModel, SEQUENCE_LABELING
from finetune.target_encoders import SequenceLabelingEncoder
from finetune.utils import indico_to_finetune_sequence, finetune_to_indico_sequence


class SequenceLabeler(BaseModel):

    def _text_to_ids_with_labels(self, X, Y=None):
        # X: list of list of text snippets.  Each snippet represents a segment of text with a consistent label
        encoder_out = self.encoder.encode_sequence_labeling(X, Y, max_length=self.config.max_length)
        seq_array = self._array_format(encoder_out.token_ids, labels=encoder_out.labels)
        return seq_array.token_ids, seq_array.mask, seq_array.labels, encoder_out.char_locs

    def _finetune(self, X, Y, batch_size=None):
        """
        X: List / array of text
        Y: Class labels
        val_size: Float fraction or int number that represents the size of the validation set.
        val_interval: The interval for which validation is performed, measured in number of steps.
        """
        train_x, train_mask, sequence_labels, _ = self._text_to_ids_with_labels(X, Y)
        return self._training_loop(train_x, train_mask, sequence_labels,
                                   batch_size=batch_size or self.config.batch_size)

    def get_target_encoder(self):
        return SequenceLabelingEncoder()

    def _text_to_ids(self, *Xs, max_length=None):
        """
        For sequence labeling this is a NOOP. See LanguageModelSequence._text_to_ids* for specific train and predict
        implementations.
        """
        return Xs

    def finetune(self, X, Y, batch_size=None):
        """
        :param X: A list of text snippets. Format: [batch_size]
        :param Y: A list of lists of annotations. Format: [batch_size, n_annotations], where each annotation is of the form:
            {'start': char_idx, 'end': char_idx, 'label': 'label'}
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        :param val_size: Float fraction or int number that represents the size of the validation set.
        :param val_interval: The interval for which validation is performed, measured in number of steps.
        """
        X, Y = indico_to_finetune_sequence(X, Y, none_value="<PAD>")
        self.target_type = SEQUENCE_LABELING
        return self._finetune(X, Y, batch_size=batch_size)

    def predict(self, X, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        doc_subseqs, _ = indico_to_finetune_sequence(X)
        x_pred, m_pred, _, token_positions = self._text_to_ids_with_labels(doc_subseqs)
        labels = self._predict(x_pred, m_pred, max_length=max_length)
        all_subseqs = []
        all_labels = []
        for text, label_seq, position_seq in zip(X, labels, token_positions):
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
                    # continue appending to current subsequencef
                    doc_subseqs[-1] += text[start_of_token:position]

                start_of_token = position
            all_subseqs.append(doc_subseqs)
            all_labels.append(doc_labels)
        doc_texts, doc_annotations = finetune_to_indico_sequence(all_subseqs, all_labels)
        return doc_annotations

    def predict_proba(self, Xs, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        raise NotImplemented  # TODO(BEN)

    def featurize(self, Xs, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(*list(zip(*Xs)), max_length=max_length)

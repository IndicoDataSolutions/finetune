import numpy as np

from finetune.lm_base import LanguageModelBase, SEQUENCE_LABELING
from finetune.target_encoders import SequenceLabelingEncoder
from finetune.utils import sequence_predict


class LanguageModelSequence(LanguageModelBase):

    def __init__(self, autosave_path, verbose=True, label_pad_target="<PAD>"):
        super().__init__(autosave_path=autosave_path, verbose=verbose)
        self.label_pad_target = label_pad_target

    def _text_to_ids_with_labels(self, *Xs):

        encoder_out = self.encoder.encode_input_sequence_labeling(*Xs, max_length=self.hparams.max_length)
        tokens, mask = self._array_format(encoder_out.token_ids)
        padded_labels = []
        for sequence in encoder_out.labels:
            padded_labels.append(sequence + (self.hparams.max_length - len(sequence)) * [self.label_pad_target])
        return tokens, mask, padded_labels, encoder_out.char_locs

    def _finetune(self, *Xs, Y, batch_size=None):
        """
        X: List / array of text
        Y: Class labels
        val_size: Float fraction or int number that represents the size of the validation set.
        val_interval: The interval for which validation is performed, measured in number of steps.
        """
        train_x, train_mask, sequence_labels, _ = self._text_to_ids_with_labels(*Xs)
        return self._training_loop(train_x, train_mask, sequence_labels,
                                   batch_size=batch_size or self.hparams.batch_size)

    def get_target_encoder(self):
        return SequenceLabelingEncoder()

    def predict_ops(self, logits):
        return sequence_predict(logits, self.predict_params)

    def _text_to_ids(self, *Xs, max_length=None):
        """
        For sequence labeling this is a NOOP. See LanguageModelSequence._text_to_ids* for specific train and predict
        implementations.
        """
        return Xs

    def finetune(self, XYs, batch_size=None):
        """
        :param XYs: An array of labeled text snippets. Format: [batch_size, sequences_per_data, snippets_per_sequence, 2]
            where the final dimension is of the format [text_snippet, label]
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        :param val_size: Float fraction or int number that represents the size of the validation set.
        :param val_interval: The interval for which validation is performed, measured in number of steps.
        """
        self.target_type = SEQUENCE_LABELING
        return self._finetune(*list(zip(*XYs)), Y=None, batch_size=batch_size)

    def predict(self, Xs, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param Xs: An iterable of lists or array of text, shape [batch, tokens]
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        sequence_major = list(zip(*Xs))
        x_pred, m_pred, _, tok_pos = self._text_to_ids_with_labels(*sequence_major)
        sparse_predictions = self._predict(x_pred, m_pred, max_length=max_length)
        output = []

        for texts, labels, token_position in zip(Xs, sparse_predictions, tok_pos):
            current_sequence = -1
            output_for_item = []
            start_of_token = float("inf")
            for lab, pos in zip(labels, token_position):
                if pos == -1:
                    continue  # Put in earlier to identify padding and meta tokens
                if pos < start_of_token:
                    # Next sequence
                    start_of_token = 0
                    output_for_item.append([])
                    current_sequence += 1
                    text = texts[current_sequence]
                if output_for_item[-1] and lab == output_for_item[-1][-1][1]:
                    output_for_item[-1][-1][0] += text[start_of_token: pos]
                else:
                    output_for_item[-1].append([text[start_of_token: pos], lab])
                start_of_token = pos
            output.append(output_for_item)
        return output

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

import itertools
import copy
import math
from collections import Counter

import tensorflow as tf
import numpy as np

from finetune.base import BaseModel, PredictMode
from finetune.encoding.target_encoders import (
    SequenceLabelingEncoder,
    SequenceMultiLabelingEncoder,
)
from finetune.nn.target_blocks import sequence_labeler
from finetune.nn.crf import sequence_decode
from finetune.encoding.sequence_encoder import (
    indico_to_finetune_sequence,
    finetune_to_indico_sequence,
)
from finetune.encoding.input_encoder import NLP
from finetune.input_pipeline import BasePipeline


class SequencePipeline(BasePipeline):
    def __init__(self, config, multi_label):
        super(SequencePipeline, self).__init__(config)
        self.multi_label = multi_label

    def _post_data_initialization(self, Y, context=None):
        Y_ = list(itertools.chain.from_iterable(Y)) if Y is not None else None
        super()._post_data_initialization(Y_, context)

    def text_to_tokens_mask(self, X, Y=None, context=None):
        pad_token = (
            [self.config.pad_token] if self.multi_label else self.config.pad_token
        )
        if context is None and self.config.use_auxiliary_info:
            context = X[1]
            X = X[0]

        out_gen = self._text_to_ids(X, Y=Y, pad_token=pad_token, context=context)
        for out in out_gen:
            if self.config.use_auxiliary_info:
                feats = {
                    "tokens": out.token_ids,
                    "mask": out.mask,
                    "context": out.context,
                }
            else:
                feats = {"tokens": out.token_ids, "mask": out.mask}

            if Y is None:
                yield feats
            if Y is not None:
                yield feats, self.label_encoder.transform(out.labels)

    def _compute_class_counts(self, encoded_dataset):
        counter = Counter()
        for doc, target_arr in encoded_dataset:
            targets = target_arr[doc["mask"].astype(np.bool)]
            counter.update(self.label_encoder.inverse_transform(targets))
        return counter

    def _format_for_encoding(self, X):
        return [X]

    def _format_for_inference(self, X):
        return [[x] for x in X]

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        target_shape = (
            [self.config.max_length, self.label_encoder.target_dim]
            if self.multi_label
            else [self.config.max_length]
        )
        if self.config.use_auxiliary_info:
            return (
                (
                    {"tokens": tf.int32, "mask": tf.float32, "context": tf.float32},
                    tf.float32,
                ),
                (
                    {
                        "tokens": TS([self.config.max_length, 2]),
                        "mask": TS([self.config.max_length]),
                        "context": TS([self.config.max_length, self.context_dim]),
                    },
                    TS(target_shape),
                ),
            )
        else:
            return (
                ({"tokens": tf.int32, "mask": tf.float32}, tf.float32),
                (
                    {
                        "tokens": TS([self.config.max_length, 2]),
                        "mask": TS([self.config.max_length]),
                    },
                    TS(target_shape),
                ),
            )

    def _target_encoder(self):
        if self.multi_label:
            return SequenceMultiLabelingEncoder()
        return SequenceLabelingEncoder()


def _combine_and_format(subtokens, start, end, raw_text):
    """
    Combine predictions on many subtokens into a single token prediction.
    Currently only valid for GPT.
    """
    result = {"start": start, "end": end}
    result["text"] = raw_text[result["start"] : result["end"]]
    probabilities = {}
    keys = subtokens[0]["probabilities"].keys()
    for k in keys:
        probabilities[k] = np.mean([token["probabilities"][k] for token in subtokens])
    result["probabilities"] = probabilities
    max_response = max(probabilities.items(), key=lambda x: x[1])
    result["label"] = max_response[0]
    result["confidence"] = max_response[1]
    return result


def _spacy_token_predictions(raw_text, tokens, probas, positions):
    """
    Go from GPT subtoken level predictions, to spacy token predictions
    """
    to_combine = []
    spacy_attn = []

    spacy_token_starts, spacy_token_ends = zip(
        *[(token.idx, token.idx + len(token.text)) for token in NLP(raw_text)]
    )
    spacy_token_idx = 0
    spacy_results = []

    for token, prob, (start, end) in zip(tokens, probas, positions):
        to_combine.append(
            {"start": start, "end": end, "token": token, "probabilities": prob}
        )

        try:
            end_match = spacy_token_ends.index(end, spacy_token_idx)
            start = spacy_token_starts[end_match]
            spacy_token_idx = end_match
        except ValueError:
            continue

        spacy_results.append(
            _combine_and_format(to_combine, start=start, end=end, raw_text=raw_text)
        )
        to_combine = []

    return spacy_results


class SequenceLabeler(BaseModel):
    """
    Labels each token in a sequence as belonging to 1 of N token classes.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    defaults = {"n_epochs": 5, "lr_warmup": 0.1}

    def __init__(self, **kwargs):
        """
        For a full list of configuration options, see `finetune.config`.

        :param config: A config object generated by `finetune.config.get_config` or None (for default config).
        :param n_epochs: defaults to `5`.
        :param lr_warmup: defaults to `0.1`,
        :param low_memory_mode: defaults to `True`,
        :param chunk_long_sequences: defaults to `True`
        :param **kwargs: key-value pairs of config items to override.
        """
        super().__init__(**kwargs)

        d = copy.deepcopy(SequenceLabeler.defaults)
        for key, value in d.items():
            if key in kwargs:
                continue
            elif key == 'n_epochs':
                value = max(self.defaults['n_epochs'], self.config.n_epochs)
            setattr(self.config, key, value)

    def _get_input_pipeline(self):
        return SequencePipeline(
            config=self.config, multi_label=self.config.multi_label_sequences
        )

    def _initialize(self):
        self.multi_label = self.config.multi_label_sequences
        return super()._initialize()

    def finetune(self, Xs, Y=None, batch_size=None):
        context = None
        if self.config.use_auxiliary_info:
            context = Xs[1]
            Xs = Xs[0]
        Xs_new, Y_new, _, _, _ = indico_to_finetune_sequence(
            Xs,
            encoder=self.input_pipeline.text_encoder,
            labels=Y,
            multi_label=self.multi_label,
            none_value=self.config.pad_token,
        )

        Y = Y_new if Y is not None else None

        if self.config.use_auxiliary_info:
            context_new = context
            Xs = [Xs_new, context_new]
        else:
            Xs = Xs_new
        return super().finetune(Xs, Y=Y, batch_size=batch_size)

    def predict(self, X, per_token=False):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param per_token: If True, return raw probabilities and labels on a per token basis
        :returns: list of class labels.
        """
        if self.config.use_auxiliary_info:
            X_with_context = copy.deepcopy(X)
            X = X[0]
        else:
            X_with_context = X
        all_subseqs = []
        all_labels = []
        all_probs = []
        all_positions = []
        chunk_size = self.config.max_length - 2
        step_size = chunk_size // 3
        doc_idx = -1
        for (
            position_seq,
            start_of_doc,
            end_of_doc,
            label_seq,
            proba_seq,
        ) in self.process_long_sequence(X_with_context):
            start, end = 0, None
            if start_of_doc:
                # if this is the first chunk in a document, start accumulating from scratch
                doc_subseqs = []
                doc_labels = []
                doc_probs = []
                doc_positions = []
                doc_starts = []

                doc_idx += 1
                start_of_token = 0
                if not end_of_doc:
                    end = step_size * 2
            else:
                if end_of_doc:
                    # predict on the rest of sequence
                    start = step_size
                else:
                    # predict only on middle third
                    start, end = step_size, step_size * 2

            label_seq = label_seq[start:end]
            position_seq = position_seq[start:end]
            proba_seq = proba_seq[start:end]

            for label, position, proba in zip(label_seq, position_seq, proba_seq):
                if position == -1:
                    # indicates padding / special tokens
                    continue

                # if there are no current subsequence
                # or the current subsequence has the wrong label
                if not doc_subseqs or label != doc_labels[-1] or per_token:
                    # start new subsequence
                    doc_subseqs.append(X[doc_idx][start_of_token:position])
                    doc_labels.append(label)
                    doc_probs.append([proba])
                    doc_positions.append((start_of_token, position))
                    doc_starts.append(start_of_token)
                else:
                    # continue appending to current subsequence
                    doc_subseqs[-1] = X[doc_idx][doc_starts[-1] : position]
                    doc_probs[-1].append(proba)
                start_of_token = position

            if end_of_doc:
                # last chunk in a document
                prob_dicts = []
                for prob_seq in doc_probs:
                    # format probabilities as dictionary
                    probs = np.mean(np.vstack(prob_seq), axis=0)
                    prob_dicts.append(
                        dict(zip(self.input_pipeline.label_encoder.classes_, probs))
                    )
                    if self.multi_label:
                        del prob_dicts[-1][self.config.pad_token]

                all_subseqs.append(doc_subseqs)
                all_labels.append(doc_labels)
                all_probs.append(prob_dicts)
                all_positions.append(doc_positions)

        _, doc_annotations = finetune_to_indico_sequence(
            raw_texts=X,
            subseqs=all_subseqs,
            labels=all_labels,
            probs=all_probs,
            none_value=self.config.pad_token,
            subtoken_predictions=self.config.subtoken_predictions,
        )

        if per_token:
            return [
                {
                    "tokens": _spacy_token_predictions(
                        raw_text=raw_text,
                        tokens=tokens,
                        probas=probas,
                        positions=positions,
                    ),
                    "prediction": predictions,
                }
                for raw_text, tokens, labels, probas, positions, predictions in zip(
                    X,
                    all_subseqs,
                    all_labels,
                    all_probs,
                    all_positions,
                    doc_annotations,
                )
            ]
        else:
            return doc_annotations

    def featurize(self, X):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X)

    def predict_proba(self, X):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :returns: list of class labels.
        """
        return self.predict(X)

    @staticmethod
    def _target_model(
        config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs
    ):
        return sequence_labeler(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            multilabel=config.multi_label_sequences,
            reuse=reuse,
            pool_idx=featurizer_state["pool_idx"],
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        trans_mats = kwargs.get("transition_matrix")
        if self.multi_label:
            logits = tf.unstack(logits, axis=-1)
            label_idxs = []
            label_probas = []
            for logits_i, trans_mat_i in zip(logits, trans_mats):
                idx, prob = sequence_decode(logits_i, trans_mat_i)
                label_idxs.append(idx)
                label_probas.append(prob[:, :, 1:])
            label_idxs = tf.stack(label_idxs, axis=-1)
            label_probas = tf.stack(label_probas, axis=-1)
        else:
            label_idxs, label_probas = sequence_decode(logits, trans_mats)
        return label_idxs, label_probas

    def _predict_proba_op(self, logits, **kwargs):
        return tf.no_op()

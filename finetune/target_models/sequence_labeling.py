import itertools
import copy
from collections import Counter

import tensorflow as tf
import numpy as np

from finetune.base import BaseModel
from finetune.encoding.target_encoders import (
    SequenceLabelingEncoder,
    SequenceMultiLabelingEncoder,
)
from finetune.nn.target_blocks import sequence_labeler
from finetune.nn.crf import sequence_decode
from finetune.encoding.sequence_encoder import (
    finetune_to_indico_sequence,
)
from finetune.encoding.input_encoder import NLP
from finetune.input_pipeline import BasePipeline
from finetune.encoding.input_encoder import tokenize_context


class SequencePipeline(BasePipeline):
    def __init__(self, config, multi_label):
        super(SequencePipeline, self).__init__(config)
        self.multi_label = multi_label

    def _post_data_initialization(self, Y):
        Y_ = list(itertools.chain.from_iterable(Y)) if Y is not None else None
        super()._post_data_initialization(Y_)

    def text_to_tokens_mask(self, X, Y=None, context=None):
        pad_token = [self.config.pad_token] if self.multi_label else self.config.pad_token
        out_gen = self._text_to_ids(X, pad_token=pad_token)
        for out in out_gen:
            feats = {"tokens": out.token_ids}
            if context is not None:
                tokenized_context = tokenize_context(context, out, self.config)
                feats['context'] = tokenized_context
            if Y is None:
                yield feats
            if Y is not None:
                min_starts = min(out.token_starts)
                max_ends = max(out.token_ends)
                filtered_labels = [
                    lab for lab in Y if lab["end"] >= min_starts and lab["start"] <= max_ends
                ]
                if self.config.filter_empty_examples and len(filtered_labels) == 0:
                    continue
                yield feats, self.label_encoder.transform(out, filtered_labels)

    def _compute_class_counts(self, encoded_dataset):
        counter = Counter()
        for doc, target_arr in encoded_dataset:
            target_arr = np.asarray(target_arr)
            decoded_targets = self.label_encoder.inverse_transform(target_arr)
            if self.multi_label:
                for label in decoded_targets:
                    counter.update(label)
            else:
                counter.update(decoded_targets)
        return counter

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = (
            [None, self.label_encoder.target_dim]
            if self.multi_label
            else [None]
        )
        return (
            (types, tf.float32,),
            (shapes, TS(target_shape),),
        )

    def _target_encoder(self):
        if self.multi_label:
            return SequenceMultiLabelingEncoder(pad_token=self.config.pad_token)
        return SequenceLabelingEncoder(pad_token=self.config.pad_token)


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

    defaults = {"add_eos_bos_to_chunk": False}

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

    def predict(self, X, per_token=False, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param per_token: If True, return raw probabilities and labels on a per token basis
        :returns: list of class labels.
        """
        all_subseqs = []
        all_labels = []
        all_probs = []
        all_positions = []
        chunk_size = self.config.max_length - 2
        step_size = chunk_size // 3
        doc_idx = -1
        for token_start_idx, token_end_idx, start_of_doc, end_of_doc, label_seq, proba_seq in self.process_long_sequence(X, context=context, **kwargs):
            if start_of_doc:
                # if this is the first chunk in a document, start accumulating from scratch
                doc_subseqs = []
                doc_labels = []
                doc_probs = []
                doc_positions = []
                doc_starts = []

                doc_idx += 1
            start, end = self.input_pipeline.chunker.useful_chunk_section(start_of_doc, end_of_doc)

            label_seq = label_seq[start:end]
            end_of_token_seq = token_end_idx[start:end]
            start_of_token_seq = token_start_idx[start:end]
            proba_seq = proba_seq[start:end]

            for label, start_idx, end_idx, proba in zip(label_seq, start_of_token_seq, end_of_token_seq, proba_seq):
                if end == -1:
                    # indicates padding / special tokens
                    continue

                # if there are no current subsequences
                # or the current subsequence has the wrong label
                if not doc_subseqs or label != doc_labels[-1] or per_token:
                    # start new subsequence
                    doc_subseqs.append(X[doc_idx][start_idx: end_idx])
                    doc_labels.append(label)
                    doc_probs.append([proba])
                    doc_positions.append((start_idx, end_idx))
                    doc_starts.append(start_idx)
                else:
                    # continue appending to current subsequence
                    doc_subseqs[-1] = X[doc_idx][doc_starts[-1]: end_idx]
                    doc_probs[-1].append(proba)

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

    def featurize(self, X, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X, **kwargs)

    def predict_proba(self, X, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :returns: list of class labels.
        """
        return self.predict(X, context=context, **kwargs)

    def _target_model(
        self, *, config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs
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
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            featurizer_state=featurizer_state,
            use_crf=self.config.crf_sequence_labeling,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        trans_mats = kwargs.get("transition_matrix")
        sequence_length = kwargs.get("sequence_length")
        if self.config.use_gpu_crf_predict.lower() == "auto":
            use_gpu_op = self.multi_label
        else:
            use_gpu_op = self.config.use_gpu_crf_predict

        if self.multi_label:
            logits = tf.unstack(logits, axis=-1)
            label_idxs = []
            label_probas = []
            for logits_i, trans_mat_i in zip(logits, trans_mats):
                idx, prob = sequence_decode(
                    logits_i, trans_mat_i, sequence_length, use_gpu_op=True, use_crf=self.config.crf_sequence_labeling
                )
                label_idxs.append(idx)
                label_probas.append(prob[:, :, 1:])
            label_idxs = tf.stack(label_idxs, axis=-1)
            label_probas = tf.stack(label_probas, axis=-1)
        else:
            label_idxs, label_probas = sequence_decode(
                logits, trans_mats, sequence_length, use_gpu_op=False, use_crf=self.config.crf_sequence_labeling
            )
        return label_idxs, label_probas

    def _predict_proba_op(self, logits, **kwargs):
        return tf.no_op()

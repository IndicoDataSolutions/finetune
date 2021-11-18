import itertools
import copy
import os
from collections import Counter, defaultdict
import math
from typing import Dict, List, Tuple, Union

import tensorflow as tf
import numpy as np

from finetune.base import BaseModel
from finetune.encoding.target_encoders import (
    SequenceLabelingEncoder,
    SequenceMultiLabelingEncoder,
)
from finetune.nn.target_blocks import sequence_labeler
from finetune.nn.crf import sequence_decode
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.encoding.input_encoder import get_spacy
from finetune.input_pipeline import BasePipeline
from finetune.util.metrics import sequences_overlap
from finetune.encoding.input_encoder import tokenize_context


class SequencePipeline(BasePipeline):
    def __init__(self, config, multi_label):
        super(SequencePipeline, self).__init__(config)
        self.multi_label = multi_label
        self.empty_counts = {"empty": 0, "labeled": 0}

    def _update_empty_ratio(self, empty):
        if empty:
            self.empty_counts["empty"] += 1
        else:
            self.empty_counts["labeled"] += 1
        return self.empty_counts

    @property
    def empty_ratio(self):
        # Smoothed to prevent zero division
        return self.empty_counts["empty"] / (self.empty_counts["labeled"] + 1)

    def text_to_tokens_mask(self, X, Y=None, context=None):
        """
        Given the text from a single document (X), and optionally the labels found
        in that document (Y), tokenize the text, and yield chunks

        If Y is provided, filter out chunks that do not contain any positive
        examples (labels) at a ratio determined by self.config.max_empty_chunk_ratio
        """
        pad_token = (
            [self.config.pad_token] if self.multi_label else self.config.pad_token
        )
        out_gen = self._text_to_ids(X, pad_token=pad_token)

        for out in out_gen:
            feats = {"tokens": out.token_ids}
            if context is not None:
                tokenized_context = tokenize_context(context, out, self.config)
                feats["context"] = tokenized_context
            if Y is None:
                yield feats
            if Y is not None:
                min_starts = min(out.token_starts)
                max_ends = max(out.token_ends)
                filtered_labels = [
                    lab
                    for lab in Y
                    if lab["end"] >= min_starts and lab["start"] <= max_ends
                ]
                empty = len(filtered_labels) == 0
                if (
                    self.config.filter_empty_examples
                    or self.empty_ratio > self.config.max_empty_chunk_ratio
                ) and empty:
                    continue
                self._update_empty_ratio(empty)
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
            [None, self.label_encoder.target_dim] if self.multi_label else [None]
        )
        return (
            (
                types,
                tf.float32,
            ),
            (
                shapes,
                TS(target_shape),
            ),
        )

    def _target_encoder(self):
        if self.multi_label:
            return SequenceMultiLabelingEncoder(pad_token=self.config.pad_token)
        return SequenceLabelingEncoder(
            pad_token=self.config.pad_token, bio_tagging=self.config.bio_tagging
        )


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
    nlp = get_spacy()

    spacy_token_starts, spacy_token_ends = zip(
        *[(token.idx, token.idx + len(token.text)) for token in nlp(raw_text)]
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


def negative_samples(preds, labels, pad="<PAD>"):
    modified_labels = []
    for p, l in zip(preds, labels):
        if isinstance(l, np.ndarray):
            l = l.tolist()
        new_labels = []
        for pi in p:
            if not any(sequences_overlap(pi, li) for li in l):
                pi["label"] = pad
                new_labels.append(pi)
        modified_labels.append(l + new_labels)
    return modified_labels


def negative_samples(preds, labels, pad="<PAD>"):
    """
    Used for auto negative sampling

    Given model predictions and ground truth labels, identify any prediction labels
    that do not exist (via sequence_overlap metric) in the ground truth, annotate
    them with the PAD label, and add them to the label set
    """
    modified_labels = []
    for p, l in zip(preds, labels):
        if isinstance(l, np.ndarray):
            l = l.tolist()
        new_labels = []
        for pi in p:
            if not any(sequences_overlap(pi, li) for li in l):
                pi["label"] = pad
                new_labels.append(pi)
        modified_labels.append(l + new_labels)
    return modified_labels


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
            elif key == "n_epochs":
                value = max(self.defaults["n_epochs"], self.config.n_epochs)
            setattr(self.config, key, value)

    def _get_input_pipeline(self):
        return SequencePipeline(
            config=self.config, multi_label=self.config.multi_label_sequences
        )

    def _initialize(self):
        self.multi_label = self.config.multi_label_sequences
        return super()._initialize()

    def finetune(
        self,
        Xs,
        Y=None,
        context=None,
        update_hook=None,
        log_hooks=None,
        X_partial=None,
        Y_partial=None,
    ):
        """
        Fit a sequence labeling model

        If X_partial and Y_partial are provided, treat those as partial data and
        Xs and Y as fully annotated data. We only perform auto negative sampling on
        fully annotated data.
        """
        if (
            self.config.chunk_long_sequences
            and self.config.auto_negative_sampling
            and Y is not None
        ):
            # clear the saver to save memory.
            self.saver.fallback  # retrieve the fallback future.
            self.saver = None
            model_copy = copy.deepcopy(self)
            if model_copy.config.tensorboard_folder:
                model_copy.config.tensorboard_folder = os.path.join(model_copy.config.tensorboard_folder, "ans_run")
            model_copy._initialize()
            model_copy.input_pipeline.total_epoch_offset = self.config.n_epochs
            self.input_pipeline.current_epoch_offset = self.config.n_epochs
            self.input_pipeline.total_epoch_offset = self.config.n_epochs

            # Train model with only positive chunks
            model_copy.config.max_empty_chunk_ratio = 0.0
            model_copy.config.auto_negative_sampling = False
            model_copy.finetune(Xs, Y=Y, context=context, update_hook=update_hook)
            initial_run_preds = []

            # Heuristic to select batch size for prediction to limit memory consumption.
            # Aim is to give us the smallest batch size that will give us full batches.
            approx_max_tokens_per_doc = max(len(x) for x in Xs) / 5
            approx_chunks_per_doc = approx_max_tokens_per_doc / (
                self.config.max_length - self.input_pipeline.chunker.total_context_width
            )
            outer_batch_size = min(
                max(int(self.config.predict_batch_size / approx_chunks_per_doc), 1),
                self.config.predict_batch_size,
            )

            with self.cached_predict():
                for b_start in range(0, len(Xs), outer_batch_size):
                    initial_run_preds += model_copy.predict(
                        Xs[b_start : b_start + outer_batch_size]
                    )
            del model_copy

            # Tag negative predictions with <PAD> label and add to label set
            Y_with_neg_samples = negative_samples(
                initial_run_preds, Y, pad=self.config.pad_token
            )

            # this means we get the same absolute number of randomly sampled empty chunks with or without this option.
            self.config.max_empty_chunk_ratio *= sum(len(yi) for yi in Y) / sum(
                len(yi) for yi in Y_with_neg_samples
            )
            Y = Y_with_neg_samples

            # Reinitialize the model including rebuilding the saver.
            self._initialize()

            # If training with partial labels, combine output of auto negative sampling on
            # gold data with partial labels data
            if X_partial and Y_partial:
                Xs = Xs + X_partial
                Y = Y + Y_partial

                # Set empty chunks ratio to 0 to remove empty chunks from partially labeled
                # data, which will only have a sample of true labels
                # TODO Determine if we need something more sophisticated for chunking
                self.config.max_empty_chunk_ratio = 0.0

        return super().finetune(
            Xs, Y=Y, context=context, update_hook=update_hook, log_hooks=log_hooks
        )

    def _pre_chunk_document(
        self, texts: List[str]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        If self.config.max_document_chars is set, "pre-chunk" any documents that
        are longer than that into multiple "sub documents" to more easily process
        large documents through prediction.

        Args:
            texts: Text of each document

        Returns:
            new_texts: Text of each document after pre chunking
            split_indices: Indices of documents that were split by pre chunking
        """
        max_doc_len = self.config.max_document_chars
        new_texts = []
        split_indices = []
        offset = 0
        # Split documents into "sub documents" if any are too long
        for doc_idx, doc in enumerate(texts):
            if len(doc) > max_doc_len:
                num_splits = math.ceil(len(doc) / max_doc_len)
                for split_idx in range(num_splits):
                    new_texts.append(
                        doc[split_idx * max_doc_len : (split_idx + 1) * max_doc_len]
                    )
                split_indices.append(
                    list(range(doc_idx + offset, doc_idx + offset + num_splits))
                )
                offset += num_splits - 1
            else:
                new_texts.append(doc)
                split_indices.append([doc_idx + offset])

        return new_texts, split_indices

    def _merge_chunked_preds(
        self,
        preds: Union[List[List[Dict]], List[Dict]],
        split_indices: List[List[int]],
        return_negative_confidence: bool = False,
    ) -> List[List[Dict]]:
        """
        If self.config.max_document_chars is set, text for long documents is split
        into multiple "sub documents". Given model predictions, and the indices
        specifying which documents have been split, join the labels for previously
        split documents together.
        preds is a list, where each element of the list corresponds to a document.
        In most cases, this will be a List[List[Dict]], where we have a List[Dict]
        for each document, which is a list of predictions.
        However, if return_negative_confidence is True, we instead have a Dict for
        each document, which contains the keys "negative_confidence" and "prediction"
        Args:
            preds: Model predictions
            split_indices: Indices specifying how documents were split
            return_negative_confidence: If True, expect preds to be List[Dict]
                instead List[List[Dict]]
        Returns:
            merged_preds: Model predictions after merging documents together
        """
        merged_preds = []
        for pred_idxs in split_indices:
            if len(pred_idxs) == 1:
                merged_preds.append(preds[pred_idxs[0]])
            # len(pred_idxs) > 1 indicates that a document was split into multiple
            # "sub documents", for which the labels need to be merged
            elif return_negative_confidence:
                doc_preds = {"prediction": []}
                all_doc_neg_confs = defaultdict(list)
                for chunk_idx, pred_idx in enumerate(pred_idxs):
                    offset = chunk_idx * self.config.max_document_chars
                    chunk_preds = preds[pred_idx]["prediction"]
                    # Add offset to label start/ends so that they index correctly
                    # into the text after joining across pre chunks
                    for i in range(len(chunk_preds)):
                        chunk_preds[i]["start"] += offset
                        chunk_preds[i]["end"] += offset
                    doc_preds["prediction"].extend(chunk_preds)
                    # Create list of negative confidences for each key, to later take the max of
                    for key, val in preds[pred_idx]["negative_confidence"].items():
                        all_doc_neg_confs[key].append(val)
                # Get max of negative conf values for each key
                doc_preds["negative_confidence"] = {
                    key: np.max(vals) for key, vals in all_doc_neg_confs.items()
                }
                merged_preds.append(doc_preds)
            else:
                doc_preds = []
                for chunk_idx, pred_idx in enumerate(pred_idxs):
                    offset = chunk_idx * self.config.max_document_chars
                    chunk_preds = preds[pred_idx]
                    # Add offset to label start/ends so that they index correctly
                    # into the text after joining across pre chunks
                    for i in range(len(chunk_preds)):
                        chunk_preds[i]["start"] += offset
                        chunk_preds[i]["end"] += offset
                    doc_preds.extend(chunk_preds)
                merged_preds.append(doc_preds)

        return merged_preds

    def predict(
        self,
        X,
        per_token=False,
        context=None,
        return_negative_confidence=False,
        **kwargs
    ):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param per_token: If True, return raw probabilities and labels on a per token basis
        :returns: list of class labels.
        """
        if self.config.max_document_chars:
            # Split each document into N "pre chunks" or length self.config.max_document_chars
            # before feeding them into the input pipeline. The split_indices will be used to
            # rejoin the labels after prediction
            X, split_indices = self._pre_chunk_document(X)

        preds = super().predict(
            X,
            per_token=per_token,
            context=context,
            return_negative_confidence=return_negative_confidence,
            **kwargs
        )

        if self.config.max_document_chars:
            preds = self._merge_chunked_preds(
                preds, split_indices, return_negative_confidence
            )

        return preds

    def _predict(
        self, zipped_data, per_token=False, return_negative_confidence=False, **kwargs
    ):
        predictions = self.process_long_sequence(zipped_data, **kwargs)
        return self._predict_decode(
            zipped_data,
            predictions,
            per_token=per_token,
            return_negative_confidence=return_negative_confidence,
            **kwargs
        )

    def _predict_decode(
        self,
        zipped_data,
        predictions,
        per_token=False,
        return_negative_confidence=False,
        **kwargs
    ):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :param per_token: If True, return raw probabilities and labels on a per token basis
        :returns: list of class labels.
        """
        classes = list(self.input_pipeline.label_encoder.classes_)
        doc_idx = -1
        doc_annotations = []
        raw_text = [data.get("raw_text", data["X"]) for data in zipped_data]
        for (
            token_start_idx,
            token_end_idx,
            start_of_doc,
            end_of_doc,
            label_seq,
            proba_seq,
            start,
            end,
        ) in predictions:
            if start_of_doc:
                # if this is the first chunk in a document, start accumulating from scratch
                doc_subseqs = []
                doc_labels = []
                doc_probs = []
                doc_positions = []
                doc_starts = []
                doc_idx += 1
                last_end = 0
                doc_level_probas = []

            label_seq = label_seq[start:end]
            end_of_token_seq = token_end_idx[start:end]
            start_of_token_seq = token_start_idx[start:end]
            proba_seq = proba_seq[start:end]

            proba_seq_masked = proba_seq.copy()

            for il, label in enumerate(label_seq):
                # covers the multilabel case where pad is not a distinct class.
                if label in classes:
                    label_idx = classes.index(label)
                    proba_seq_masked[il:, label_idx] = 0.0
            doc_level_probas.append(np.max(proba_seq_masked, axis=0))

            for label, start_idx, end_idx, proba in zip(
                label_seq, start_of_token_seq, end_of_token_seq, proba_seq
            ):
                if end_idx == -1:
                    # indicates padding / special tokens
                    continue

                assert start_idx >= last_end, "Start idx: {}, last_end: {}".format(
                    start_idx, last_end
                )
                last_end = end_idx

                if self.config.group_bio_tagging:
                    group_prefix = ""
                    if label[:3] == "BG-" or label[:3] == "IG-":
                        group_prefix, label = label[:3], label[3:]
                if self.config.bio_tagging:
                    bio_prefix = None
                    if label != self.config.pad_token:
                        bio_prefix, label = label[:2], label[2:]
                if self.config.group_bio_tagging and label != self.config.pad_token:
                    # Save the group prefix so the grouping target models can
                    # extract grouping information down the line
                    label = group_prefix + label

                def _get_label(label):
                    if label[:3] == "BG-" or label[:3] == "IG-":
                        return label[3:]
                    return label

                # if there are no current subsequences
                # or the current subsequence has the wrong label
                # or bio tagging is on and we have a B- tag
                if (
                    not doc_subseqs
                    or per_token
                    or (self.config.bio_tagging and bio_prefix == "B-")
                    or (self.config.group_bio_tagging and group_prefix == "BG-")
                    or (
                        label != doc_labels[-1]
                        and (
                            # Merge spans if the labels are the same,
                            # disregarding group BIO tags
                            # This is safe as we already hard break on BG-, so
                            # we will only be merging IG- tags
                            not self.config.group_bio_tagging
                            or _get_label(label) != _get_label(doc_labels[-1])
                        )
                    )
                ):
                    assert start_idx <= end_idx, "Start: {}, End: {}".format(
                        start_idx, end_idx
                    )
                    # start new subsequence
                    doc_subseqs.append(raw_text[doc_idx][start_idx:end_idx])
                    doc_labels.append(label)
                    doc_probs.append([proba])
                    doc_positions.append((start_idx, end_idx))
                    doc_starts.append(start_idx)
                else:
                    assert start_idx <= end_idx, "Start: {}, End: {}".format(
                        start_idx, end_idx
                    )
                    # continue appending to current subsequence
                    assert doc_starts[-1] <= end_idx, "Start: {}, End: {}".format(
                        doc_starts[-1], end_idx
                    )
                    doc_subseqs[-1] = raw_text[doc_idx][doc_starts[-1] : end_idx]
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

                _, doc_annotations_sample = finetune_to_indico_sequence(
                    raw_texts=[raw_text[doc_idx]],
                    subseqs=[doc_subseqs],
                    labels=[doc_labels],
                    probs=[prob_dicts],
                    none_value=self.config.pad_token,
                    subtoken_predictions=self.config.subtoken_predictions,
                    bio_tagging=self.config.bio_tagging
                    or self.config.group_bio_tagging,
                )
                if per_token:
                    doc_annotations.append(
                        {
                            "tokens": _spacy_token_predictions(
                                raw_text=raw_text[doc_idx],
                                tokens=doc_subseqs,
                                probas=prob_dicts,
                                positions=doc_positions,
                            ),
                            "prediction": doc_annotations_sample[0],
                        }
                    )
                elif return_negative_confidence:
                    doc_annotations.append(
                        {
                            "prediction": doc_annotations_sample[0],
                            "negative_confidence": dict(
                                zip(classes, np.max(doc_level_probas, axis=0))
                            ),
                        }
                    )

                else:
                    doc_annotations.append(doc_annotations_sample[0])
        return doc_annotations

    def featurize(self, X, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param Xs: An iterable of lists or array of text, shape [batch, n_inputs, tokens]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X, **kwargs)

    def predict_proba(
        self, X, context=None, return_negative_confidence=False, **kwargs
    ):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :returns: list of class labels.
        """
        return self.predict(
            X,
            context=context,
            return_negative_confidence=return_negative_confidence,
            **kwargs
        )

    def _target_model(
        self,
        *,
        config,
        featurizer_state,
        targets,
        n_outputs,
        train=False,
        reuse=None,
        **kwargs
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
                    logits_i,
                    trans_mat_i,
                    sequence_length,
                    use_gpu_op=True,
                    use_crf=self.config.crf_sequence_labeling,
                )
                label_idxs.append(idx)
                label_probas.append(prob[:, :, 1:])
            label_idxs = tf.stack(label_idxs, axis=-1)
            label_probas = tf.stack(label_probas, axis=-1)
        else:
            label_idxs, label_probas = sequence_decode(
                logits,
                trans_mats,
                sequence_length,
                use_gpu_op=False,
                use_crf=self.config.crf_sequence_labeling,
            )
        return label_idxs, label_probas

    def _predict_proba_op(self, logits, **kwargs):
        return tf.no_op()

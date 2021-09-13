import json
import warnings
from collections import Counter

import numpy as np
import tensorflow as tf

from finetune.model import PredictMode
from finetune.nn.crf import sequence_decode
from finetune.encoding.input_encoder import tokenize_context
from finetune.util.imbalance import compute_class_weights
from finetune.target_models.sequence_labeling import (
    SequencePipeline,
    SequenceLabeler,
)
from finetune.target_models.seq2seq import (
    S2SPipeline,
    HFS2S,
)
from finetune.encoding.group_target_encoders import (
    SequenceLabelingEncoder,
    GroupSequenceLabelingEncoder,
    MultiCRFGroupSequenceLabelingEncoder,
    PipelineSequenceLabelingEncoder,
    BROSEncoder,
    JointBROSEncoder,
    GroupRelationEncoder,
    JointGroupRelationEncoder,
    SequenceLabelingTextEncoder,
    GroupLabelingTextEncoder,
    JointLabelingTextEncoder,
)
from finetune.nn.group_target_blocks import (
    multi_crf_group_labeler,
    multi_logit_group_labeler,
    bros_decoder,
    joint_bros_decoder,
    group_relation_decoder,
    joint_group_relation_decoder,
)

class GroupingPipeline(SequencePipeline):
    """
    Base grouping pipeline.

    Just removes autonegative sampling from text_to_token_mask, as it breaks
    when passing labels of the form (labels, groups)
    """
    def __init__(self, config):
        super().__init__(config, multi_label=False)

    def text_to_tokens_mask(self, X, Y=None, context=None):
        pad_token = self.config.pad_token
        out_gen = self._text_to_ids(X, pad_token=pad_token)

        for out in out_gen:
            feats = {"tokens": out.token_ids}
            if context is not None:
                tokenized_context = tokenize_context(context, out, self.config)
                feats["context"] = tokenized_context
            if Y is None:
                yield feats
            if Y is not None:
                yield feats, self.label_encoder.transform(out, Y)

class JointGroupingPipeline(GroupingPipeline):
    """
    Pipeline for models that jointly tag and group with two seperate tasks.

    Implements _compute_class_counts that calls inverse_transform() with
    only_labels=True, so we can compute class weights for the NER tagging task.
    """
    def _compute_class_counts(self, encoded_dataset):
        counter = Counter()
        for doc, target_arr in encoded_dataset:
            target_arr = np.asarray(target_arr)
            # Only labels flag gives us only NER labels
            decoded_targets = self.label_encoder.inverse_transform(target_arr,
                                                                   only_labels=True)
            counter.update(decoded_targets)
        return counter


class GroupTaggingPipeline(GroupingPipeline):
    def _target_encoder(self):
        return GroupSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                            bio_tagging=self.config.bio_tagging)

class MultiCRFPipeline(JointGroupingPipeline):
    def _target_encoder(self):
        return MultiCRFGroupSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                                    bio_tagging=self.config.bio_tagging)

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [2, None]
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

class PipelinePipeline(GroupingPipeline):
    def __init__(self, config, group=True):
        super().__init__(config)
        self.group = group

    def _target_encoder(self):
        return PipelineSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                               bio_tagging=self.config.bio_tagging,
                                               group=self.group)

class BROSPipeline(GroupingPipeline):
    def _target_encoder(self):
        return BROSEncoder(pad_token=self.config.pad_token)

    def _compute_class_counts(self, encoded_dataset):
        start_counter = Counter()
        next_counter = Counter()
        for doc, target_arr in encoded_dataset:
            target_arr = np.asarray(target_arr)
            decoded_targets = self.label_encoder.inverse_transform(target_arr)

            start_tokens, next_tokens = decoded_targets
            # We only care about 0 vs non-0 for weighting next tokens
            # STOP and CONT are hardcoded values in BROSEncoder
            next_tokens = ["STOP" if n == 0 else "CONT" for n in next_tokens]

            # {"<PAD>": count, "GROUP": count}
            start_counter.update(start_tokens)
            # {"STOP": count, "CONT": count}
            next_counter.update(next_tokens)
        # Passed directly to and unpacked in _compute_class_weights
        return (start_counter, next_counter)

    def _compute_class_weights(self, class_weights, class_counts):
        start_counter, next_counter = class_counts
        start_weights = compute_class_weights(
            class_weights=class_weights, class_counts=start_counter
        )
        next_weights = compute_class_weights(
            class_weights=class_weights, class_counts=next_counter
        )
        # {"<PAD>": ratio, "GROUP": ratio, "STOP": ratio, "CONT": ratio}
        return {**start_weights, **next_weights}

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [2, None]
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

class JointBROSPipeline(JointGroupingPipeline):
    def _target_encoder(self):
        return JointBROSEncoder(pad_token=self.config.pad_token,
                                bio_tagging=self.config.bio_tagging)

    def _compute_class_counts(self, encoded_dataset):
        ner_counter = Counter()
        start_counter = Counter()
        next_counter = Counter()
        for doc, target_arr in encoded_dataset:
            # TODO: Get rid of code duplication
            target_arr = np.asarray(target_arr)
            decoded_targets = self.label_encoder.inverse_transform(target_arr)

            ner_labels, start_tokens, next_tokens = decoded_targets
            # We only care about 0 vs non-0 for weighting next tokens
            # STOP and CONT are hardcoded values in BROSEncoder
            next_tokens = ["STOP" if n == 0 else "CONT" for n in next_tokens]

            ner_counter.update(ner_labels)
            start_counter.update(start_tokens)
            next_counter.update(next_tokens)
        # Passed directly to and unpacked in _compute_class_weights
        return (ner_counter, start_counter, next_counter)

    def _compute_class_weights(self, class_weights, class_counts):
        ner_counter, start_counter, next_counter = class_counts
        ner_weights = compute_class_weights(
            class_weights=class_weights, class_counts=ner_counter
        )
        start_weights = compute_class_weights(
            class_weights=class_weights, class_counts=start_counter
        )
        next_weights = compute_class_weights(
            class_weights=class_weights, class_counts=next_counter
        )
        # The tensor this creates is unpacked in the target block
        return {**ner_weights, **start_weights, **next_weights}

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [3, None]
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

class GroupRelationPipeline(GroupingPipeline):
    def _target_encoder(self):
        return GroupRelationEncoder(pad_token=self.config.pad_token,
                                    n_groups=self.config.n_groups)

    def _compute_class_counts(self, encoded_dataset):
        counter = Counter()
        for doc, target_arr in encoded_dataset:
            target_arr = np.asarray(target_arr)
            label = self.label_encoder.inverse_transform(target_arr)

            # Count number of tokens in non-pad groups and pad group
            groups = label[:-1]
            counter["GROUP"] += sum([sum(g) for g in groups])
            pad_group = label[-1]
            counter["PAD"] += sum(pad_group)
        return counter

    def _compute_class_weights(self, class_weights, class_counts):
        weights = compute_class_weights(
            class_weights=class_weights, class_counts=class_counts
        )
        return weights

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [self.config.n_groups, None]
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

class JointGroupRelationPipeline(JointGroupingPipeline):
    def _target_encoder(self):
        return JointGroupRelationEncoder(pad_token=self.config.pad_token,
                                         n_groups=self.config.n_groups)

    def _compute_class_counts(self, encoded_dataset):
        ner_counter = Counter()
        group_counter = Counter()
        for doc, target_arr in encoded_dataset:
            target_arr = np.asarray(target_arr)
            ner_labels, group_labels = self.label_encoder.inverse_transform(
                target_arr, class_weights=True
            )
            ner_counter.update(ner_labels)
            # Count number of tokens in non-pad groups and pad group
            non_pad_groups = group_labels[:-1]
            group_counter["GROUP"] += sum([sum(g) for g in non_pad_groups])
            pad_group = group_labels[-1]
            group_counter["PAD"] += sum(pad_group)
        # Unpacked in _compute_class_weights
        return (ner_counter, group_counter)

    def _compute_class_weights(self, class_weights, class_counts):
        ner_counts, group_counts = class_counts
        ner_weights = compute_class_weights(
            class_weights=class_weights, class_counts=ner_counts,
        )
        group_weights = compute_class_weights(
            class_weights=class_weights, class_counts=group_counts,
        )
        return {**ner_weights, **group_weights}

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [self.config.n_groups, None]
        return (
            (
                types,
                {
                    "groups": tf.float32,
                    "tags": tf.float32,
                }
            ),
            (
                shapes,
                {
                    "groups": TS(target_shape),
                    "tags": TS([None]),
                }
            ),
        )
    
class SequenceS2SPipeline(S2SPipeline):
    def _target_encoder(self):
        return SequenceLabelingTextEncoder(self, self.config.s2s_decoder_max_length)

class GroupS2SPipeline(S2SPipeline):
    def _target_encoder(self):
        return GroupLabelingTextEncoder(self, self.config.s2s_decoder_max_length)

class JointS2SPipeline(S2SPipeline):
    def _target_encoder(self):
        return JointLabelingTextEncoder(self, self.config.s2s_decoder_max_length)


class GroupSequenceLabeler(SequenceLabeler):
    """
    Group sequence labeler.

    Simplest implementation of grouping. Encodes group information into labels
    with a BIO style tagging scheme, and extracts group information post
    sequence labeling decoding.
    """
    defaults = {"group_bio_tagging": True, "bio_tagging": True}

    def _get_input_pipeline(self):
        return GroupTaggingPipeline(
            config=self.config,
        )

    def _predict(
        self, zipped_data, per_token=False, return_negative_confidence=False, **kwargs
    ):
        # We expect the returned annotations to be of standard Finetune form,
        # with entities collected into their individual spans and with grouping
        # information preserved in the labels

        # Note that as an side effect of how we segement spans, groups starting
        # in the middle of entities will break entities into multiple pieces
        # (This effect is minimized when BIO tagging is set, as the model will
        # never see a BG-I- tag during training)

        annotations = super()._predict(zipped_data, per_token=per_token,
                                       return_negative_confidence=return_negative_confidence,
                                       **kwargs);

        # Extract group information from labels
        all_groups = []
        for data, labels in zip(zipped_data, annotations):
            groups = []
            text = data["X"]
            for label in labels:
                if label["label"][:3] != "BG-" and label["label"][:3] != "IG-":
                    continue
                pre, tag = label["label"][:3], label["label"][3:]

                # Remove group information from NER label
                label["label"] = tag

                if (not groups or (pre == "BG-")):
                    groups.append({
                        "spans": [
                            {
                                "start": label["start"],
                                "end": label["end"],
                                "text": label["text"],
                            }
                        ],
                        "label": None
                    })
                else:
                    # Note that we extend groups to include the next IG- span,
                    # regardless of if there is space between the end of the
                    # previous span and the start of the IG- span.
                    last_token = groups[-1]["spans"][-1]
                    last_token["end"] = label["end"]
                    last_token["text"] = text[last_token["start"]:last_token["end"]]
            all_groups.append(groups)

        return list(zip(annotations, all_groups))

class MultiCRFGroupSequenceLabeler(GroupSequenceLabeler):
    """
    Multi-CRF group sequence labeler.

    Implements the same BIO tagging style grouping method employed by the Group
    Sequence Labeler, but utilizes two CRFS - one for NER labels and one for
    group labels. Uses a custom target block and predict op, but, as the target
    encoder converts the predictions to the same form as the Group Sequence
    Labeler, inherits the predict function.
    """
    def _get_input_pipeline(self):
        return MultiCRFPipeline(
            config=self.config,
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
        return multi_crf_group_labeler(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        trans_mats = kwargs.get("transition_matrix")
        group_trans_mats = kwargs.get("group_transition_matrix")
        sequence_length = kwargs.get("sequence_length")
        if self.config.use_gpu_crf_predict.lower() == "auto":
            use_gpu_op = False
        else:
            use_gpu_op = self.config.use_gpu_crf_predict

        with tf.compat.v1.variable_scope("tag_sequence_decode"):
            idxs, probas = sequence_decode(
                logits[0],
                trans_mats,
                sequence_length,
                use_gpu_op=False,
                use_crf=self.config.crf_sequence_labeling,
            )
        with tf.compat.v1.variable_scope("group_sequence_decode"):
            group_idxs, group_probas = sequence_decode(
                logits[1],
                group_trans_mats,
                sequence_length,
                use_gpu_op=False,
                use_crf=self.config.crf_sequence_labeling,
            )

        # Produces [batch_size, 2, seq_len]
        # Required to pass predictions through Finetune
        idxs = tf.stack([idxs, group_idxs], axis=1)

        # Broadcast probabilities to make [batch, seq_len, n_classes * 3] matrix
        batch_seq_shape, n_classes = tf.shape(probas)[:2], tf.shape(probas)[-1]
        final_shape = tf.concat((batch_seq_shape, [n_classes * 3]), 0)
        # [batch, seq_len, n_classes, 1] * [batch, seq_len, 1, 3] =
        # [batch, seq_len, n_classes, 3]
        probas = tf.expand_dims(probas, 3) * tf.expand_dims(group_probas, 2)
        # Reshape to [batch, seq_len, n_classes * 3]
        probas = tf.reshape(probas, final_shape)

        return idxs, probas

class MultiLogitGroupSequenceLabeler(GroupSequenceLabeler):
    """
    Multi-logit group sequence labeler.

    Implements the same BIO tagging style grouping method employed by the Group
    Sequence Labeler, but generates logits for the NER task and the grouping
    task, then broadcasts them together to create the final logits. As the
    logits returned by the target block are of the same form as the Group
    Sequence Labeler, all other methods are inherited.
    """
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
        return multi_logit_group_labeler(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            **kwargs
        )

class PipelineSequenceLabeler(SequenceLabeler):
    """
    Pipeline group sequence labeler.

    Serves as a easy way to train either end of a pipeline method. With the
    group attribute set, the target encoder encodes group information as NER
    tags, and the predict function transforms it back to group format. With the
    group attribute not set, it functions almost identically to a Sequence
    Labeler, without auto negative sampling. Note that both sides always take
    both label and group information for convenience.
    """
    defaults = {"bio_tagging": True}
    def __init__(self, group=True, **kwargs):
        self.group = group
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return PipelinePipeline(
            config=self.config,
            group=self.group
        )

    def _predict(
        self, zipped_data, per_token=False, return_negative_confidence=False, **kwargs
    ):
        # Transform NER span labels to group format
        annotations = super()._predict(zipped_data, per_token=per_token,
                                       return_negative_confidence=return_negative_confidence,
                                       **kwargs);
        # TODO: Fix this breakig when loading from model
        if not self.group:
            return annotations

        all_groups = []
        for labels in annotations:
            groups = []
            for label in labels:
                groups.append({
                    "spans": [{
                        "start": label["start"],
                        "end": label["end"],
                        "text": label["text"],
                    }],
                    "label": None
                })
            all_groups.append(groups)
        return all_groups

class BROSLabeler(SequenceLabeler):
    """
    BROS labeler. See https://openreview.net/pdf?id=punMXQEsPr0 for details.

    Implements a BROS style decoder. Intended for use in a pipeline method, so
    will only return group information. However, takes in both label in
    group information for convenience.
    """
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return BROSPipeline(config=self.config)

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
        return bros_decoder(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        # [Batch Size, Sequence Length, 2]
        start_token_logits = logits["start_token_logits"]
        start_token_idxs = tf.argmax(start_token_logits, axis=-1)
        start_token_probas = tf.nn.softmax(start_token_logits, axis=-1)

        # [Batch Size, Sequence Length, Sequence Length + 1]
        next_token_logits = logits["next_token_logits"]
        next_token_idxs = tf.argmax(next_token_logits, axis=-1)
        next_token_probas = tf.nn.softmax(next_token_logits, axis=-1)

        # Produces [batch_size, 2, seq_len]
        idxs = tf.stack([start_token_idxs, next_token_idxs], axis=1)
        # Unclear how we want to represent confidences, this is a placeholder
        probas = start_token_probas

        return idxs, probas

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        predictions = list(self.process_long_sequence(zipped_data))
        return self._predict_decode(zipped_data, predictions, **kwargs)

    def _predict_decode(self, zipped_data, predictions, **kwargs):
        raw_texts = list(data.get("raw_text", data["X"]) for data in zipped_data)
        all_groups = []
        for text_idx, (
            token_start_idx,
            token_end_idx,
            start_of_doc,
            end_of_doc,
            label_seq,
            proba_seq,
            start,
            end,
        ) in enumerate(predictions):
            # BROS must operate on the entire input sequence, no chunks allowed
            assert start_of_doc and end_of_doc, "Chunk found in BROS!"

            start_tokens, next_tokens = label_seq

            text = raw_texts[text_idx]
            doc_groups = []
            for i, (label, start_idx, end_idx) in enumerate(zip(
                start_tokens, token_start_idx, token_end_idx
            )):
                if label == self.config.pad_token:
                    continue
                
                # New group for each start token
                group_spans = [{
                    "start": start_idx,
                    "end": end_idx,
                    "text": text[start_idx:end_idx],
                }] 
                
                # Iteratively build the rest of the group
                group_idxs = [i]
                current_idx = next_tokens[i] 
                while current_idx > 0:
                    # Avoid getting caught in a loop of next tokens
                    if current_idx in group_idxs:
                        warnings.warn("Cylical group found!")
                        break
                    if current_idx >= len(token_start_idx):
                        # Happens when the end token is the next token
                        warnings.warn("OOB Index found!")
                        break
                    # No guarentee that tokens point in sequential order, so we
                    # check all existing spans for adjacency
                    # Continuous spans are required as part of the group format
                    current_start_idx = token_start_idx[current_idx]
                    current_end_idx = token_end_idx[current_idx]
                    for span in group_spans:
                        if (current_start_idx >= span["end"] and
                            not text[span["end"]:current_start_idx].strip()):
                            span["end"] = current_end_idx
                            span["text"] = text[span["start"]:span["end"]]
                            break
                        elif (current_end_idx <= span["start"] and
                              not text[current_end_idx:span["start"]].strip()):
                            span["start"] = current_start_idx
                            span["text"] = text[span["start"]:span["end"]]
                            break
                    else:
                        group_spans.append({
                            "start": current_start_idx,
                            "end": current_end_idx,
                            "text": text[current_start_idx:current_end_idx],
                        })

                    group_idxs.append(current_idx)
                    current_idx = next_tokens[current_idx]
                # Sort spans to match group format (could be moved to eval)
                group_spans = sorted(group_spans, key=lambda x: x["start"])
                doc_groups.append({
                    "spans": group_spans,
                    "label": None,
                })
            # Sort groups to match group format (could be moved to eval)
            doc_groups = sorted(doc_groups, key=lambda x: x["spans"][0]["start"])
            all_groups.append(doc_groups)
        return all_groups

class JointBROSLabeler(BROSLabeler, SequenceLabeler):
    """
    Joint BROS labeler. See BROSLabaler or SequenceLabeler for detail on
    respective target models.

    Implements both a standard Sequence Labeler and a BROS labeler. Each
    decoder acts independently but share a featurizer. Is a joint model, and
    therefore returns both NER and group information.
    """
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return JointBROSPipeline(config=self.config)

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
        return joint_bros_decoder(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        group_idxs, group_probas = BROSLabeler._predict_op(
            self, logits, **kwargs
        )
        ner_idxs, ner_probas = SequenceLabeler._predict_op(
            self, logits["ner_logits"], **kwargs
        )

        # Produces [batch_size, 3, seq_len]
        # Required to pass predictions through Finetune
        ner_idxs = tf.cast(ner_idxs, tf.int32)
        group_idxs = tf.cast(group_idxs, tf.int32)
        start_token_idxs, next_token_idxs = group_idxs[:, 0, :], group_idxs[:, 1, :]
        idxs = tf.stack([ner_idxs, start_token_idxs, next_token_idxs], axis=1)
        probas = ner_probas

        return idxs, probas

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        # Seperate out predictions for each model
        # This is somewhat horrifying
        predictions = list(self.process_long_sequence(zipped_data))
        (token_start_idx, token_end_idx, start_of_doc, end_of_doc, label_seq,
         proba_seq, start, end) = list(zip(*predictions))
        ner_labels, start_token_labels, next_token_labels = list(zip(*label_seq))

        group_labels = list(zip(start_token_labels, next_token_labels))
        group_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, group_labels, proba_seq, start, end)
        ner_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, ner_labels, proba_seq, start, end)


        group_predictions = BROSLabeler._predict_decode(
            self, zipped_data, group_predictions, **kwargs
        )
        ner_predictions = SequenceLabeler._predict_decode(
            self, zipped_data, ner_predictions, **kwargs
        )

        return list(zip(ner_predictions, group_predictions))

class GroupRelationLabeler(SequenceLabeler):
    """
    Group relation labeler. See https://arxiv.org/pdf/2011.01675v2.pdf for details

    Implements a group relation decoder. Intended for use in a pipeline method,
    so will only return group information. However, takes in both label in
    group information for convenience.
    """
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return GroupRelationPipeline(config=self.config)

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
        return group_relation_decoder(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_groups=self.config.n_groups,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            hidden_size=config.group_hidden_size,
            num_attention_heads=config.group_attention_heads,
            n_layers=config.group_n_layers,
            query_size=config.relation_hidden_size,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        # [batch_size, seq_len, n_groups]
        logits = tf.transpose(logits, perm=[0, 2, 1])
        probs = tf.nn.softmax(logits, axis=-1)
        # [batch_size, seq_len]
        probs = tf.compat.v1.Print(probs, [tf.shape(logits), tf.shape(probs)])
        token_groups = tf.argmax(probs, axis=-1)
        return token_groups, probs

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        predictions = list(self.process_long_sequence(zipped_data))
        return self._predict_decode(zipped_data, predictions, **kwargs)

    def _predict_decode(self, zipped_data, predictions, **kwargs):
        raw_texts = list(data.get("raw_text", data["X"]) for data in zipped_data)
        all_groups = []
        for text_idx, (
            token_start_idx,
            token_end_idx,
            start_of_doc,
            end_of_doc,
            label_seq,
            proba_seq,
            start,
            end,
        ) in enumerate(predictions):
            assert start_of_doc and end_of_doc, "Chunk found in group relation!!"

            text = raw_texts[text_idx]

            # The first token is always padding
            pad_group = label_seq[0]
            
            doc_groups = {}
            for token_group, token_start, token_end in zip(label_seq,
                                                           token_start_idx,
                                                           token_end_idx):
                if token_group == pad_group or token_end == -1:
                    continue
                group = doc_groups.get(token_group, {
                    "spans": [],
                    "label": None
                })

                # If there are no tokens or if there is text between the last
                # token and the current one, start a new span
                if (not group["spans"] or
                    text[group["spans"][-1]["end"]:token_start].strip()):
                    group["spans"].append({
                        "start": token_start,
                        "end": token_end,
                        "text": text[token_start:token_end]
                    })
                # Otherwise, continue span
                else:
                    last_span = group["spans"][-1]
                    last_span["end"] = token_end
                    last_span["text"] = text[last_span["start"]:
                                             last_span["end"]]

                doc_groups[token_group] = group
            doc_groups = list(doc_groups.values())
            all_groups.append(doc_groups)
        return all_groups

class JointGroupRelationLabeler(GroupRelationLabeler, SequenceLabeler):
    """
    Joint group relation labeler. See GroupRelationLabeler or SequenceLabeler
    for detail on respective target models.

    Implements both a standard Sequence Labeler and a group relation labeler.
    Each decoder acts independently but share a featurizer. Is a joint model,
    and therefore returns both NER and group information.
    """
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return JointGroupRelationPipeline(config=self.config)

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
        return joint_group_relation_decoder(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            n_groups=self.config.n_groups,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            hidden_size=config.group_hidden_size,
            num_attention_heads=config.group_attention_heads,
            n_layers=config.group_n_layers,
            query_size=config.relation_hidden_size,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        group_idxs, group_probas = GroupRelationLabeler._predict_op(
            self, logits["group_logits"]
        )
        ner_idxs, ner_probas = SequenceLabeler._predict_op(
            self, logits["ner_logits"], **kwargs
        )

        # Stack to feed predictions as a single tensor
        # Will be unpacked in the target encoder
        # [batch_size, 2, seq_len]
        group_idxs = tf.cast(group_idxs, tf.int32)
        idxs = tf.stack([group_idxs, ner_idxs], axis=1)
        probas = ner_probas

        return idxs, probas

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        # Seperate out predictions for each model
        predictions = list(self.process_long_sequence(zipped_data))
        (token_start_idx, token_end_idx, start_of_doc, end_of_doc, label_seq,
         proba_seq, start, end) = list(zip(*predictions))
        ner_labels, group_labels = list(zip(*label_seq))

        group_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, group_labels, proba_seq, start, end)
        ner_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, ner_labels, proba_seq, start, end)


        group_predictions = GroupRelationLabeler._predict_decode(
            self, zipped_data, group_predictions, **kwargs
        )
        ner_predictions = SequenceLabeler._predict_decode(
            self, zipped_data, ner_predictions, **kwargs
        )

        return list(zip(ner_predictions, group_predictions))

class SequenceS2S(HFS2S):
    """
   T5 wrapper for sequence labeling.
    """
    def _get_input_pipeline(self):
        return SequenceS2SPipeline(self.config)

    def _predict(self, zipped_data, **kwargs):
        raw_texts = list(data.get("raw_text", data["X"]) for data in zipped_data)
        preds = self._inference(zipped_data, predict_keys=[PredictMode.NORMAL],  **kwargs)
        return self.input_pipeline.label_encoder.inverse_transform(preds, raw_texts)

class GroupS2S(HFS2S):
    """
    T5 wrapper for grouping.
    """
    def _get_input_pipeline(self):
        return GroupS2SPipeline(self.config)

    def _predict(self, zipped_data, **kwargs):
        raw_texts = list(data.get("raw_text", data["X"]) for data in zipped_data)
        preds = self._inference(zipped_data, predict_keys=[PredictMode.NORMAL],  **kwargs)
        return self.input_pipeline.label_encoder.inverse_transform(preds, raw_texts)

class JointS2S(HFS2S):
    """
    T5 wrapper for joint sequence tagging and grouping.
    """
    def _get_input_pipeline(self):
        return JointS2SPipeline(self.config)

    def _predict(self, zipped_data, **kwargs):
        raw_texts = list(data.get("raw_text", data["X"]) for data in zipped_data)
        preds = self._inference(zipped_data, predict_keys=[PredictMode.NORMAL],  **kwargs)
        return self.input_pipeline.label_encoder.inverse_transform(preds, raw_texts)



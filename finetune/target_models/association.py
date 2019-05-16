import itertools
import tensorflow as tf
import numpy as np

from finetune.base import BaseModel
from finetune.encoding.target_encoders import SequenceLabelingEncoder
from finetune.nn.target_blocks import association
from finetune.nn.crf import sequence_decode
from finetune.encoding.sequence_encoder import indico_to_finetune_sequence, finetune_to_indico_sequence
from finetune.input_pipeline import BasePipeline
from finetune.errors import FinetuneError
from finetune.base import LOGGER
from finetune.model import PredictMode


class AssociationPipeline(BasePipeline):
    def __init__(self, config, multi_label):
        super(AssociationPipeline, self).__init__(config)
        self.multi_label = multi_label
        self.association_encoder = SequenceLabelingEncoder()
        self.association_encoder.fit(config.association_types + [self.config.pad_token])
        self.association_pad_idx = self.association_encoder.transform([self.config.pad_token])

    def _post_data_initialization(self, Y):
        Y_ = list(itertools.chain.from_iterable([y[0] for y in Y]))
        super()._post_data_initialization(Y_)

    def text_to_tokens_mask(self, X, Y=None):
        pad_token = [self.config.pad_token] if self.multi_label else self.config.pad_token
        if Y is not None:
            Y = list(zip(*Y))
        out_gen = self._text_to_ids(X, Y=Y, pad_token=(pad_token, pad_token, -1, -2))
        class_list = self.association_encoder.classes_.tolist()
        assoc_pad_id = class_list.index(pad_token)
        for out in out_gen:
            feats = {"tokens": out.token_ids, "mask": out.mask}
            if Y is None:
                yield feats
            else:
                labels = []
                assoc_mat = [[assoc_pad_id for _ in range(len(out.labels))] for _ in range(len(out.labels))]
                for i, (l, _, _, idx) in enumerate(out.labels):
                    labels.append(l)
                    for j, (_, a_t, a_i, _) in enumerate(out.labels):
                        if a_t != pad_token and idx == a_i:
                            assoc_mat[i][j] = class_list.index(a_t)

                yield feats, {"labels": self.label_encoder.transform(labels),
                              "associations": np.array(assoc_mat, dtype=np.int32)}

    def _format_for_encoding(self, X):
        return [X]

    def _format_for_inference(self, X):
        return [[x] for x in X]

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        target_shape = (
            [self.config.max_length, self.label_encoder.target_dim]
            if self.multi_label else [self.config.max_length]
        )
        return (
            (
                {
                    "tokens": tf.int32,
                    "mask": tf.float32
                },
                {
                    "labels": tf.int32,
                    "associations": tf.int32
                }
            ),
            (
                {
                    "tokens": TS([self.config.max_length, 2]),
                    "mask": TS([self.config.max_length])
                },
                {
                    "labels": TS(target_shape),
                    "associations": TS([self.config.max_length, self.config.max_length])
                }
            )
        )

    def _target_encoder(self):
        return SequenceLabelingEncoder()


class Association(BaseModel):
    """
    Labels each token in a sequence as belonging to 1 of N token classes and then builds a set of edges
    between the labeled edges.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def __init__(self, **kwargs):
        """
        For a full list of configuration options, see `finetune.config`.

        :param n_epochs: defaults to `5`.
        :param lr_warmup: defaults to `0.1`,
        :param low_memory_mode: defaults to `True`,
        :param chunk_long_sequences: defaults to `True`
        :param **kwargs: key-value pairs of config items to override.
        """
        self.multi_label = False
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return AssociationPipeline(config=self.config, multi_label=False)

    def _initialize(self):
        if self.config.multi_label_sequences:
            raise FinetuneError("Multi label association not supported")
        return super()._initialize()

    def finetune(self, Xs, Y=None, batch_size=None):
        """
        :param Xs: A list of strings.
        :param Y: A list of labels of the same format as sequence labeling but with an option al additional field
        of the form:
        ```
            {
                ...
                "association":{
                        "index": a,
                        "relationship": relationship_name
                }
                ...
        ```
        where index is the index of the relationship target into the label list and relationship_name is the type of
        the relationship.
        """
        if self.config.association_types is None:
            raise FinetuneError("Please set config.association_types before calling finetune.")
        Xs, Y_new, association_type, association_idx, idxs = indico_to_finetune_sequence(
            Xs,
            encoder=self.input_pipeline.text_encoder,
            labels=Y,
            multi_label=False,
            none_value=self.config.pad_token
        )

        Y = list(zip(Y_new, association_type, association_idx, idxs)) if Y is not None else None
        return super().finetune(Xs, Y=Y, batch_size=batch_size)

    def prune_probs(self, prob_matrix, labels):
        viable_edges = self.config.viable_edges
        association_types = list(self.input_pipeline.association_encoder.classes_)
        if viable_edges is None:
            return prob_matrix

        for i, l1 in enumerate(labels):
            if l1 not in viable_edges:
                prob_matrix[i, :, :] = 0.0
                continue

            elif None not in viable_edges[l1]:
                prob_matrix[i, :, self.input_pipeline.association_pad_idx] = 0.0

            for cls in association_types:
                for j, l2 in enumerate(labels):
                    if l1 not in viable_edges or l2 not in [c_t[0] for c_t in viable_edges[l1] if
                                                            c_t and c_t[1] == cls]:
                        prob_matrix[i, j, association_types.index(cls)] = 0.0  # this edge doesnt fit the schema
                        
        return prob_matrix

    def predict(self, X):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: A list / array of text, shape [batch]
        :returns: list of class labels.
        """
        pad_token = [self.config.pad_token] if self.multi_label else self.config.pad_token
        if self.config.viable_edges is None:
            LOGGER.warning("config.viable_edges is not set, this is probably incorrect.")

        #TODO(Ben) combine this into the sequence labeling model??

        chunk_size = self.config.max_length - 2
        step_size = chunk_size // 3
        arr_encoded = list(itertools.chain.from_iterable(
            self.input_pipeline._text_to_ids([x], pad_token=(pad_token, pad_token, -1, -2))
            for x in X
        ))
        lens = [len(a.char_locs) for a in arr_encoded]
        labels, batch_probas, associations = [], [], []
        predict_keys = [
            PredictMode.SEQUENCE,
            PredictMode.SEQUENCE_PROBAS,
            PredictMode.ASSOCIATION,
            PredictMode.ASSOCIATION_PROBAS
        ]
        for l, pred in zip(lens, self._inference(X, predict_keys=predict_keys)):
            pred_labels = self.input_pipeline.label_encoder.inverse_transform(pred[PredictMode.SEQUENCE])
            pred_labels = [label if i < l else self.config.pad_token for i, label in enumerate(pred_labels)]
            labels.append(pred_labels)
            batch_probas.append(pred[PredictMode.SEQUENCE_PROBAS])
            pred["association_probs"] = self.prune_probs(pred[PredictMode.ASSOCIATION_PROBAS], pred_labels)
            most_likely_associations, most_likely_class_id = zip(
                *[np.unravel_index(np.argmax(a, axis=None), a.shape) for a in pred[PredictMode.ASSOCIATION_PROBAS]]
            )
            associations.append((
                most_likely_associations,
                self.input_pipeline.association_encoder.inverse_transform(most_likely_class_id),
                [
                    prob[idx, cls] for prob, idx, cls in zip(
                        pred["association_probs"],
                        most_likely_associations,
                        most_likely_class_id
                    )
                ]
            ))
        all_subseqs = []
        all_labels = []
        all_probs = []
        all_assocs = []

        doc_idx = -1
        for chunk_idx, (label_seq, proba_seq, association) in enumerate(zip(labels, batch_probas, associations)):
            association_idx, association_class, association_prob = association

            position_seq = arr_encoded[chunk_idx].char_locs
            start_of_doc = arr_encoded[chunk_idx].token_ids[0][0] == self.input_pipeline.text_encoder.start
            end_of_doc = (
                    chunk_idx + 1 >= len(arr_encoded) or
                    arr_encoded[chunk_idx + 1].token_ids[0][0] == self.input_pipeline.text_encoder.start
            )
            start, end = 0, None
            if start_of_doc:
                # if this is the first chunk in a document, start accumulating from scratch
                doc_subseqs = []
                doc_labels = []
                doc_probs = []
                doc_assocs = []
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

            for tok_idx, (label, position, proba) in enumerate(zip(label_seq, position_seq, proba_seq)):
                if position == -1:
                    # indicates padding / special tokens
                    continue

                # if there are no current subsequence
                # or the current subsequence has the wrong label
                if not doc_subseqs or label != doc_labels[-1]:
                    # start new subsequence
                    doc_subseqs.append(X[doc_idx][start_of_token:position])
                    doc_labels.append(label)
                    doc_probs.append([proba])
                    doc_assocs.append(
                        [
                            (tok_idx, association_idx[tok_idx], association_class[tok_idx], association_prob[tok_idx])
                        ]
                    )
                else:
                    # continue appending to current subsequence
                    doc_subseqs[-1] += X[doc_idx][start_of_token:position]
                    doc_probs[-1].append(proba)
                    doc_assocs[-1].append(
                        (
                            tok_idx, association_idx[tok_idx], association_class[tok_idx], association_prob[tok_idx]
                        )
                    )

                start_of_token = position

            if end_of_doc:
                # last chunk in a document
                prob_dicts = []
                for prob_seq in doc_probs:
                    # format probabilities as dictionary
                    probs = np.mean(np.vstack(prob_seq), axis=0)
                    prob_dicts.append(dict(zip(self.input_pipeline.label_encoder.classes_, probs)))
                    if self.multi_label:
                        del prob_dicts[-1][self.config.pad_token]

                all_subseqs.append(doc_subseqs)
                all_labels.append(doc_labels)
                all_probs.append(prob_dicts)
                all_assocs.append(doc_assocs)

        _, doc_annotations = finetune_to_indico_sequence(
            raw_texts=X,
            encoder=self.input_pipeline.text_encoder,
            subseqs=all_subseqs,
            labels=all_labels,
            probs=all_probs,
            associations=all_assocs,
            subtoken_predictions=self.config.subtoken_predictions,
            none_value=self.config.pad_token
        )
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
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return association(
            hidden=featurizer_state['sequence_features'],
            pool_idx=featurizer_state['pool_idx'],
            targets=targets,
            n_targets=n_outputs,
            config=config,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):

        associations = logits["association"]
        logits = logits["sequence"]

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

        association_prob = tf.nn.softmax(associations, axis=-1)
        association_pred = tf.argmax(associations, axis=-1)

        return (
            {
                PredictMode.SEQUENCE: label_idxs,
                PredictMode.ASSOCIATION: association_pred
            },
            {
                PredictMode.SEQUENCE_PROBAS: label_probas,
                PredictMode.ASSOCIATION_PROBAS: association_prob
            }
        )

    def _predict_proba_op(self, logits, **kwargs):
        return tf.no_op()

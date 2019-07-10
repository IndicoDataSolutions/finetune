import itertools
import copy
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle

from finetune.base import BaseModel
from finetune.encoding.target_encoders import OneHotLabelEncoder
from finetune.nn.target_blocks import classifier
from finetune.input_pipeline import BasePipeline
from finetune.model import PredictMode
from finetune.base_models.gpt.encoder import finetune_to_indico_explain


class ClassificationPipeline(BasePipeline):

    def resampling(self, Xs, Y):
        if self.config.oversample:
            idxs, Ys = shuffle(*RandomOverSampler().fit_sample([[i] for i in range(len(Xs))], Y))
            return [Xs[i[0]] for i in idxs], Ys
        return Xs, Y

    def _target_encoder(self):
        return OneHotLabelEncoder()


class Classifier(BaseModel):
    """ 
    Classifies a single document into 1 of N categories.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    defaults = {
        "low_memory_mode": True,
        "chunk_long_sequences": True
    }

    def __init__(self, **kwargs):
        d = copy.deepcopy(Classifier.defaults)
        d.update(kwargs)
        super().__init__(**d)

    def _get_input_pipeline(self):
        return ClassificationPipeline(self.config)

    def featurize(self, X):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X)

    def predict(self, X, probas=False):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        chunk_size = self.config.max_length - 2
        step_size = chunk_size // 3
        arr_encoded = list(itertools.chain.from_iterable(self.input_pipeline._text_to_ids(x) for x in X))
        labels, batch_probas = [], []
        for pred in self._inference(X, predict_keys=[PredictMode.PROBAS, PredictMode.NORMAL], n_examples=len(arr_encoded)):
            label = self.input_pipeline.label_encoder.inverse_transform([pred[PredictMode.NORMAL]])
            label = label.pop()
            labels.append(label)
            batch_probas.append(pred[PredictMode.PROBAS])

        all_subseqs = []
        all_labels = []
        all_probs = []
        all_positions = []

        doc_idx = -1
        for chunk_idx, (label, proba) in enumerate(zip(labels, batch_probas)):
            position_seq = arr_encoded[chunk_idx].char_locs
            start_of_doc = arr_encoded[chunk_idx].token_ids[0][0] == self.input_pipeline.text_encoder.start
            end_of_doc = (
                    chunk_idx + 1 >= len(arr_encoded) or
                    arr_encoded[chunk_idx + 1].token_ids[0][0] == self.input_pipeline.text_encoder.start
            )
            """
            Chunk idx for prediction.  Dividers at `step_size` increments.
            [  1  |  1  |  2  |  3  |  3  ]
            """
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

            doc_labels.append(label)
            doc_probs.append(proba)

            if end_of_doc:
                # last chunk in a document
                print('')
                print(doc_probs)
                mean_pool = np.mean(doc_probs, axis=0)
                print(mean_pool)
                pred = np.argmax(mean_pool)
                one_hot = np.zeros_like(mean_pool)
                one_hot[pred] = 1
                label = self.input_pipeline.label_encoder.inverse_transform([one_hot])
                print(label)
                print('')
                all_labels.append(label)
                all_probs.append(mean_pool)

        if probas:
            return all_probs
        else:
            print(all_probs)
            return all_labels

    def predict_proba(self, X):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        #return super().predict_proba(X)
        return self.predict(X, proba=True)

    def finetune(self, X, Y=None, batch_size=None):
        """
        :param X: list or array of text.
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=Y, batch_size=batch_size)

    @classmethod
    def get_eval_fn(cls):
        return lambda labels, targets: np.mean(np.asarray(labels) == np.asarray(targets))

    @staticmethod
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        if "explain_out" in featurizer_state:
            shape = tf.shape(featurizer_state['explain_out']) # batch, seq, hidden
            flat_explain = tf.reshape(featurizer_state['explain_out'], [shape[0] * shape[1], shape[2]])
            hidden = tf.concat((featurizer_state["features"],flat_explain), 0)
        else:
            hidden = featurizer_state["features"]

        clf_out = classifier(
            hidden=hidden,
            targets=targets, 
            n_targets=n_outputs, 
            config=config,
            train=train,
            reuse=reuse,
            **kwargs
        )
        if "explain_out" in featurizer_state:
            logits = clf_out["logits"]
            clf_out["logits"] = logits[: shape[0]]
            clf_out["explanation"] = tf.nn.softmax(
                tf.reshape(
                    logits[shape[0]:], tf.concat((shape[:2], [tf.shape(logits)[-1]]), 0)
                ),
                -1)
        return clf_out

    def explain(self, Xs):
        explanation = self._inference(Xs, predict_keys=[PredictMode.EXPLAIN, PredictMode.NORMAL])
        classes = self.input_pipeline.label_encoder.target_labels
        out = []
        bases = []
        preds = []
        for values in explanation:
            explain_sample = values[PredictMode.EXPLAIN]
            preds.append(values[PredictMode.NORMAL])
            out.append(explain_sample[1:])
            bases.append(explain_sample[0])
        processed = finetune_to_indico_explain(Xs, out, self.input_pipeline.text_encoder, attention=False)

        for base, sample, cls in zip(bases, processed, preds):
            weights = sample['explanation']
            weights = np.array([base] + weights[:-1]) - weights
            n_classes = weights.shape[-1]
            norm = np.max([np.abs(np.max(weights, 0)), abs(np.min(weights, 0))], 0) * n_classes
            explanation = weights / norm + 1 / n_classes

            sample['explanation'] = {c: explanation[:, i] for i, c in enumerate(classes)}
            sample["prediction"] = self.input_pipeline.label_encoder.inverse_transform([cls])[0]
            
        return processed

    def _predict_op(self, logits, **kwargs):
        return tf.contrib.seq2seq.hardmax(logits)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

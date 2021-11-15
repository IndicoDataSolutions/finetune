from warnings import warn
import itertools
import copy
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.utils import shuffle

from finetune.base import BaseModel
from finetune.encoding.target_encoders import OneHotLabelEncoder, NoisyLabelEncoder
from finetune.nn.target_blocks import classifier
from finetune.input_pipeline import BasePipeline
from finetune.model import PredictMode
from finetune.base_models.gpt.encoder import finetune_to_indico_explain


class ClassificationPipeline(BasePipeline):

    def _target_encoder(self):
        return OneHotLabelEncoder()

class NoisyClassificationPipeline(BasePipeline):

    def _target_encoder(self):
        return NoisyLabelEncoder()

class Classifier(BaseModel):
    """
    Classifies a single document into 1 of N categories.
    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return ClassificationPipeline(self.config)

    def featurize(self, X, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.
        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X, **kwargs)

    def predict(self, X, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.
        :param X: list or array of text to embed.
        :returns: list of class labels.

        Chunk idx for prediction.  Dividers at `step_size` increments.
        [  1  |  1  |  2  |  3  |  3  ]
        """
        return super().predict(X, context=context, **kwargs)

    def _predict(self, zipped_data, probas=False, **kwargs):
        all_labels = []
        all_probs = []
        doc_probs = []
        for _,  _, start_of_doc, end_of_doc, _, proba, _, _ in self.process_long_sequence(zipped_data, **kwargs):
            start, end = 0, None
            doc_probs.append(proba)

            if end_of_doc:
                # last chunk in a document
                mean_pool = np.mean(doc_probs, axis=0)
                pred = np.argmax(mean_pool)
                one_hot = np.zeros_like(mean_pool)
                one_hot[pred] = 1
                label = self.input_pipeline.label_encoder.inverse_transform([one_hot])
                label = np.squeeze(label).tolist()
                all_labels.append(label)
                all_probs.append(mean_pool)
                doc_probs = []

        if probas:
            return all_probs
        else:
            assert len(all_labels) == len(zipped_data)
            return np.asarray(all_labels)

    def _predict_proba(self, zipped_data, **kwargs):
        """
        Produces a probability distribution over classes for each example in X.
        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        # this is simply a vector of probabilities, not a dict from classes to class probs
        return self._predict(zipped_data, probas=True, **kwargs)

    def finetune(self, X, Y=None, context=None, **kwargs):
        """
        :param X: list or array of text.
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=Y, context=context, **kwargs)

    @classmethod
    def get_eval_fn(cls):
        return lambda labels, targets: np.mean(
            np.asarray(labels) == np.asarray(targets)
        )

    def _target_model(
        self, *, config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs
    ):
        if "explain_out" in featurizer_state:
            shape = tf.shape(input=featurizer_state["explain_out"])  # batch, seq, hidden
            flat_explain = tf.reshape(
                featurizer_state["explain_out"], [shape[0] * shape[1], shape[2]]
            )
            hidden = tf.concat((featurizer_state["features"], flat_explain), 0)
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
                    logits[shape[0] :],
                    tf.concat((shape[:2], [tf.shape(input=logits)[-1]]), 0),
                ),
                -1,
            )
        return clf_out

    def explain(self, Xs, context=None):
        zipped_data = self.input_pipeline.zip_list_to_dict(X=Xs, context=context)
        explanation = self._inference(
            zipped_data, predict_keys=[PredictMode.EXPLAIN, PredictMode.NORMAL]
        )
        classes = self.input_pipeline.label_encoder.target_labels
        out = []
        bases = []
        preds = []
        for values in explanation:
            explain_sample = values[PredictMode.EXPLAIN]
            preds.append(values[PredictMode.NORMAL])
            out.append(explain_sample[1:])
            bases.append(explain_sample[0])
        processed = finetune_to_indico_explain(
            Xs, out, self.input_pipeline.text_encoder, attention=False
        )

        for base, sample, cls in zip(bases, processed, preds):
            weights = sample["explanation"]
            weights = np.array([base] + weights[:-1]) - weights
            n_classes = weights.shape[-1]
            norm = (
                np.max([np.abs(np.max(weights, 0)), abs(np.min(weights, 0))], 0)
                * n_classes
            )
            explanation = weights / norm + 1 / n_classes

            sample["explanation"] = {
                c: explanation[:, i] for i, c in enumerate(classes)
            }
            sample["prediction"] = self.input_pipeline.label_encoder.inverse_transform(
                [cls]
            )[0]

        return processed

    def _predict_op(self, logits, **kwargs):
        return tfa.seq2seq.hardmax(logits)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

class NoisyClassifier(Classifier, BaseModel):
    """
    Classifies a single document into 1 of N categories, taking a probability
    distribution over the labels as input (a dictionary with float values).

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def _get_input_pipeline(self):
        return NoisyClassificationPipeline(self.config)

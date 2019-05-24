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

    def _get_input_pipeline(self):
        return ClassificationPipeline(self.config)

    def featurize(self, X):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X)

    def predict(self, X):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        return super().predict(X)

    def predict_proba(self, X):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return super().predict_proba(X)

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

            sample['explanation'] = {c: explanation[i, :] for i, c in enumerate(classes)}
            sample["prediction"] = self.input_pipeline.label_encoder.inverse_transform([cls])[0]
            
        return processed

    def _predict_op(self, logits, **kwargs):
        return tf.argmax(logits, -1)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

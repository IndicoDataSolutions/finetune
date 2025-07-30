import numpy as np
import tensorflow as tf

from finetune.base import BaseModel
from finetune.encoding.target_encoders import MultilabelClassificationEncoder
from finetune.input_pipeline import BasePipeline
from finetune.nn.target_blocks import MultiClassifier as MultiClassifierBlock
from finetune.util.imbalance import compute_class_weights


class MultilabelClassificationPipeline(BasePipeline):
    def _compute_class_weights(self, class_weights, class_counts):
        class_weights = compute_class_weights(
            class_weights=class_weights,
            class_counts=class_counts,
            n_total=self.config.dataset_size,
            multilabel=True,
        )
        return class_weights

    def _target_encoder(self):
        return MultilabelClassificationEncoder()


class MultiLabelClassifier(BaseModel):
    """
    Classifies a single document into up to N of N categories.

    Implemented via a sum of N sigmoid losses applied a linear projection of the base model's output representation.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_input_pipeline(self):
        return MultilabelClassificationPipeline(self.config)

    def _get_threshold(self, threshold):
        if threshold is None:
            return self.config.multi_label_threshold
        return threshold

    def featurize(self, X, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X, **kwargs)

    def predict(self, X, threshold=None, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        return super().predict(X, threshold=threshold, context=context, **kwargs)

    def _predict(self, zipped_data, threshold=None, probas=False, **kwargs):
        threshold = self._get_threshold(threshold)
        all_labels = []
        for pred_bundle in self.process_long_sequence(zipped_data, **kwargs):
            if pred_bundle["start_of_doc"]:
                # if this is the first chunk in a document, start accumulating from scratch
                doc_probs = []

            doc_probs.append(pred_bundle["probas"])

            if pred_bundle["end_of_doc"]:
                # last chunk in a document
                means = np.mean(doc_probs, axis=0)
                if probas:
                    all_labels.append(means)
                else:
                    label = self.input_pipeline.label_encoder.inverse_transform(
                        np.expand_dims(means, 0) > threshold
                    )[0]
                    all_labels.append(list(label))
        return all_labels

    def _predict_proba(self, zipped_data, **kwargs):
        return self._predict(zipped_data, threshold=None, probas=True, **kwargs)

    def finetune(self, X, Y=None, context=None, **kwargs):
        """
        :param X: list or array of text.
        :param Y: A list of lists containing labels for the corresponding X
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=Y, context=context, **kwargs)

    def target_block(self, *, config, n_outputs, **kwargs):
        return MultiClassifierBlock(
            n_targets=n_outputs,
            dropout_rate=config.clf_p_drop,
            renorm_after_class_weights=config.renorm_after_class_weights,
            threshold=config.multi_label_threshold,
            name="model",
            **kwargs
        )

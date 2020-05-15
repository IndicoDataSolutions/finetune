import numpy as np
import tensorflow as tf
import copy

from finetune.errors import FinetuneError
from finetune.base import BaseModel
from finetune.target_models.classifier import Classifier, ClassificationPipeline
from finetune.encoding.input_encoder import  EncodedOutput, tokenize_context


class ComparisonPipeline(ClassificationPipeline):
    def _format_for_encoding(self, X):
        return X

    def _text_to_ids(self, pair, pad_token=None):
        """
        Format comparison examples as a list of IDs

        pairs: Array of text, shape [batch, 2]
        """
        assert (
            self.config.chunk_long_sequences is False
        ), "Chunk Long Sequences is not compatible with comparison"
        arr_forward = next(super()._text_to_ids(pair))
        reversed_pair = pair[::-1]
        arr_backward = next(super()._text_to_ids(reversed_pair))
        kwargs = arr_forward._asdict()
        kwargs["tokens"] = [arr_forward.tokens, arr_backward.tokens]
        kwargs["token_ids"] = np.stack(
            [arr_forward.token_ids, arr_backward.token_ids], 0
        )
        yield EncodedOutput(**kwargs)

    def text_to_tokens_mask(self, X, Y=None, context=None):
        out_gen = self._text_to_ids(X, pad_token=self.config.pad_token)
        for i, out in enumerate(out_gen):
            if context is None:
                feats = {"tokens": out.token_ids}
            else:
                out_forward = EncodedOutput(
                    token_ids=out.token_ids[0],
                    tokens=out.token_ids[0],
                    token_ends=out.token_ends,
                    token_starts=out.token_starts,
                )
                out_backward = EncodedOutput(
                    token_ids=out.token_ids[1],
                    tokens=out.token_ids[1],
                    token_ends=out.token_ends,
                    token_starts=out.token_starts,
                )
                tokenized_context_forward = tokenize_context(context[0], out_forward, self.config)
                tokenized_context_backward = tokenize_context(context[1], out_backward, self.config)
                tokenized_context = [tokenized_context_forward, tokenized_context_backward]
                feats = {"tokens": out.token_ids, "context": tokenized_context}
            if Y is None:
                yield feats
            else:
                yield feats, self.label_encoder.transform([Y])[0]

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {
            "tokens": TS([2, None])
        }
        if self.config.use_auxiliary_info:
            TS = tf.TensorShape
            types["context"] = tf.float32
            shapes["context"] = TS([2, None, self.config.context_dim])
        return (
            (types, tf.float32,),
            (shapes, TS([self.target_dim]),),
        )


class Comparison(Classifier):
    """
    Compares two documents to solve a classification task.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    defaults = {"chunk_long_sequences": False}

    def __init__(self, **kwargs):
        d = copy.deepcopy(Comparison.defaults)
        d.update(kwargs)
        super().__init__(**kwargs)
        if self.config.chunk_long_sequences:
            raise FinetuneError(
                "Multifield model is incompatible with chunk_long_sequences = True in config."
            )

    def _get_input_pipeline(self):
        return ComparisonPipeline(self.config)

    def _pre_target_model_hook(self, featurizer_state):
        featurizer_state["sequence_features"] = tf.abs(
            tf.reduce_sum(input_tensor=featurizer_state["sequence_features"], axis=1)
        )
        featurizer_state["features"] = tf.abs(
            tf.reduce_sum(input_tensor=featurizer_state["features"], axis=1)
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
        return super(Comparison, self)._target_model(
            config=config,
            featurizer_state=featurizer_state,
            targets=targets,
            n_outputs=n_outputs,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def predict(self, pairs, context=None, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.


        :param pairs: Array of text, shape [batch, 2]
        :returns: list of class labels.
        """
        return BaseModel.predict(self, pairs, context=context, **kwargs)

    def predict_proba(self, pairs, context=None, **kwargs):
        """
        Produces a probability distribution over classes for each example in X.


        :param pairs: Array of text, shape [batch, 2]
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return BaseModel.predict_proba(self, pairs, context=context, **kwargs)

    def featurize(self, pairs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param pairs: Array of text, shape [batch, 2]
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return BaseModel.featurize(self, pairs)

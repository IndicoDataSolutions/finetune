import tensorflow as tf
from tensorflow import TensorShape as TS
import numpy as np

from finetune.errors import FinetuneError
from finetune.base import PredictMode, BaseModel
from finetune.base_models import RoBERTa, BERT
from finetune.input_pipeline import BasePipeline
from finetune.nn.target_blocks import masked_language_model


class MaskedLanguageModelPipeline(BasePipeline):
    
    def _target_encoder(self):
        pass

    def feed_shape_type_def(self):
        return (
            (
                {
                    "tokens": tf.int32,
                    "mask": tf.float32,
                    "mlm_weights": tf.float32,
                    "mlm_ids": tf.int32,
                    "mlm_positions": tf.int32
                },
            ),
            (
                {
                    "tokens": TS([None, 2]),
                    "mask": TS([None]),
                    "mlm_weights": TS([self.config.max_masked_tokens * self.config.batch_size]),
                    "mlm_ids": TS([self.config.max_masked_tokens * self.config.batch_size]),
                    "mlm_positions": TS([self.config.max_masked_tokens * self.config.batch_size]),
                },
            ),
        )

    def text_to_tokens_mask(self, X, Y=None):
        out_gen = self._text_to_ids(X, pad_token=self.config.pad_token)
       
        for out in out_gen:
            seq_len = out.token_ids.shape[0]
            mlm_mask = np.random.rand(seq_len) < self.config.mask_proba
            mask_type = np.random.choice(
                ["mask", "random", "unchanged"], 
                size=seq_len, 
                p=[0.8, 0.1, 0.1]
            )
            random_tokens = np.random.randint(0, self.text_encoder.vocab_size, size=seq_len)

            # Make sure we don't accidentally mask the start / separator / end token
            mlm_mask[
                np.isin(
                    out.token_ids[:, 0],
                    [
                        self.text_encoder.start_token, 
                        self.text_encoder.delimiter_token,
                        self.text_encoder.end_token
                    ]
                )
            ] = False 
            mlm_ids = out.token_ids[:, 0][mlm_mask]
            expected_length = self.config.max_masked_tokens * self.config.batch_size
            pad_size = expected_length - len(mlm_ids)
            mlm_weights = np.pad(np.ones_like(mlm_ids), [(0, pad_size)], constant_values=0., mode="constant")
            mlm_ids = np.pad(mlm_ids, [(0, pad_size)], constant_values=0, mode="constant")
            mlm_positions = np.pad(np.where(mlm_mask)[0], [(0, pad_size)], constant_values=0, mode="constant")


            out.token_ids[:, 0][mlm_mask & (mask_type == 'mask')] = self.text_encoder.mask_token
            out.token_ids[:, 0][mlm_mask & (mask_type == 'random')] = random_tokens[mlm_mask & (mask_type == 'random')]

            feats = {
                "tokens": out.token_ids, 
                "mask": out.mask,
                "mlm_weights": mlm_weights,
                "mlm_ids": mlm_ids,
                "mlm_positions": mlm_positions
            }
            yield feats


class MaskedLanguageModel(BaseModel):
    """ 
    A Masked Language Model for Finetune

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not issubclass(self.config.base_model, (BERT, RoBERTa)):
            raise FinetuneError("MLM training is currently only supported for BERT and RoBERTa base models.")

    def _get_input_pipeline(self):
        return MaskedLanguageModelPipeline(self.config)

    def predict(self, *args, **kwargs):
        """
        Not supported by `MaskedLanguageModel`
        """
        raise FinetuneError(
            "The `{}` class does not support calling `model.predict()`".format(
                self.__class__.__name__
            )
        )

    def predict_proba(self, *args, **kwargs):
        """
        Not supported by `MaskedLanguageModel`
        """
        raise FinetuneError(
            "The `{}` class does not support calling `model.predict_proba()`".format(
                self.__class__.__name__
            )
        )

    def _predict_op(self):
        pass

    def _predict_proba_op(self):
        pass

    def finetune(self, X, batch_size=None, **kwargs):
        """
        :param X: list or array of text.
        :param mask_proba: the likelihood of masking a subtoken.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=None, batch_size=batch_size, **kwargs)

    @staticmethod
    def _target_model(
        config, featurizer_state, n_outputs, train=False, reuse=None, **kwargs
    ):
        return masked_language_model(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            multilabel=config.multi_label_sequences,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            **kwargs
        )

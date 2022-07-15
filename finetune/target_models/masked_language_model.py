from multiprocessing.sharedctypes import Value
import tensorflow as tf
from tensorflow import TensorShape as TS
import numpy as np

from finetune.errors import FinetuneError
from finetune.base import PredictMode, BaseModel
from finetune.base_models import RoBERTa, BERT
from finetune.input_pipeline import BasePipeline
from finetune.nn.target_blocks import masked_language_model, sequence_labeler_low_shot
from finetune.encoding.target_encoders import (
    SequenceLabelingEncoder,
)

class MaskedLanguageModelPipeline(BasePipeline):
    
    def _target_encoder(self):
        return SequenceLabelingEncoder(
            pad_token=self.config.pad_token, bio_tagging=self.config.bio_tagging
        )
    def feed_shape_type_def(self):
        return (
            (
                {
                    "tokens": tf.int32,
                    "mlm_weights": tf.float32,
                    "mlm_ids": tf.int32,
                    "mlm_positions": tf.int32
                },
            ),
            (
                {
                    "tokens": TS([None]),
                    "mlm_weights": TS([None]),
                    "mlm_ids": TS([None]),
                    "mlm_positions": TS([None]),
                },
            ),
        )

    def text_to_tokens_mask(self, X, Y=None):
        out_gen = self._text_to_ids(X, pad_token=self.config.pad_token)
       
        for out in out_gen:
            seq_len = out.token_ids.shape[0]
            mlm_positions = None
            retries = -1
            while mlm_positions is None or len(mlm_positions) == 0 and retries < 10:
                retries += 1
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
                        out.token_ids,
                        [
                            self.text_encoder.start_token, 
                            self.text_encoder.delimiter_token,
                            self.text_encoder.end_token
                        ]
                    )
                ] = False
                mlm_positions = np.where(mlm_mask)[0]
            if retries == 10:
                print("Retried 10 times and couln't get any masked examples, original lenght = {}".format(seq_len))
                continue
            
            if len(mlm_positions) > self.config.max_masked_tokens: # subsample
                np.random.shuffle(mlm_positions) # means we don't bias the begining of the sequence
                mlm_positions = mlm_positions[:self.config.max_masked_tokens]
                mlm_mask = np.zeros_like(mlm_mask)
                mlm_mask[mlm_positions] = True
                
            mlm_ids = out.token_ids[mlm_mask]

            
            out.token_ids[mlm_mask & (mask_type == 'mask')] = self.text_encoder.mask_token
            out.token_ids[mlm_mask & (mask_type == 'random')] = random_tokens[mlm_mask & (mask_type == 'random')]

            
            if Y is None:
                yield {
                    "tokens": out.token_ids, 
                    "mlm_weights": np.ones_like(mlm_ids),
                    "mlm_ids": mlm_ids,
                    "mlm_positions": mlm_positions
                }
            else:
                min_starts = min(out.token_starts)
                max_ends = max(out.token_ends)
                filtered_labels = [
                    lab
                    for lab in Y
                    if lab["end"] >= min_starts and lab["start"] <= max_ends
                ]
                seq_labeling_labels = self.label_encoder.transform(out, filtered_labels)
                extra_positions = [i for i, v in enumerate(seq_labeling_labels) if v > 0]
                extra_ids = [v + 50005 for v in seq_labeling_labels if v > 0] # 50005 is roberta specific
                new_mlm_ids = np.concatenate((mlm_ids, extra_ids))
                if len(new_mlm_ids) == 0:
                    raise ValueError()
                yield {
                    "tokens": out.token_ids, 
                    "mlm_weights": np.ones_like(new_mlm_ids),
                    "mlm_ids": new_mlm_ids,
                    "mlm_positions": np.concatenate((mlm_positions, extra_positions))

                }

class MaskedLanguageModel(BaseModel):
    """ 
    A Masked Language Model for Finetune

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    defaults = dict(low_memory_mode=True, batch_size=48, xla=True)

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

    def finetune(self, X, Y=None, **kwargs):
        """
        :param X: list or array of text.
        :param mask_proba: the likelihood of masking a subtoken.
        """
        return super().finetune(X, Y=Y, force_build_lm=True, **kwargs)

    @staticmethod
    def _target_model(
        config, featurizer_state, n_outputs, train=False, reuse=None, **kwargs
    ):
        return None

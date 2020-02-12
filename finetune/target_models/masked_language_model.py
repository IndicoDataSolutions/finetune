import tensorflow as tf
from tensorflow import TensorShape as TS
import numpy as np

from finetune.errors import FinetuneError
from finetune.base import PredictMode, BaseModel
from finetune.base_models import RoBERTa, BERT
from finetune.input_pipeline import BasePipeline
from finetune.nn.target_blocks import masked_language_model
from finetune.encoding.input_encoder import tokenize_context
from finetune.encoding.input_encoder import EncodedOutput
from tensorflow.data import Dataset


class MaskedLanguageModelPipeline(BasePipeline):

    def _target_encoder(self):
        pass

    def feed_shape_type_def(self):
        types = {
                "tokens": tf.int32,
                "mask": tf.float32,
                "mlm_weights": tf.float32,
                "mlm_ids": tf.int32,
                "mlm_positions": tf.int32
            }
        shapes = {
                "tokens": TS([None, 2]),
                "mask": TS([None]),
                "mlm_weights": TS([self.config.max_masked_tokens * self.config.batch_size]),
                "mlm_ids": TS([self.config.max_masked_tokens * self.config.batch_size]),
                "mlm_positions": TS([self.config.max_masked_tokens * self.config.batch_size]),
            }
        types, shapes = self._add_context_info_if_present(types, shapes)
        return (
            (types,),
            (shapes,),
        )

    def text_to_tokens_mask(self, X, Y=None, context=None):
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
            if context:
                try:
                    tokenized_context = tokenize_context(context, out, self.config)
                    feats['context'] = tokenized_context
                except:
                    print('Failure in context alignment for: ')
                    print(out.tokens)
                    print(context)
                    continue
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

    def predict_top_k_report(self, input_text, k=5, context=None, **kwargs):
        """
        Only works for a single example at a time.

        Args:
        - input_text (str)
        - context (list of dicts)

        """
        prediction_info = self._inference(
                [input_text],
                predict_keys=[
                    PredictMode.GENERATE_TEXT,
                    PredictMode.MLM_IDS,
                    PredictMode.MLM_POSITIONS],
                context=[context],
                force_build_lm=True,
                **kwargs)

        predicted_tokens = [{'prediction_ids': [self.input_pipeline.text_encoder.decode([i]) for i in pred['GEN_TEXT'][-k:][::-1]],
                             'original_token_id': self.input_pipeline.text_encoder.decode([pred['MLM_IDS']]),
                             'position': pred['MLM_POSITIONS']} for pred in prediction_info]
        mask_positions = [i['position'] for i in predicted_tokens]

        tokens = self.input_pipeline.text_encoder._encode([input_text]).tokens[0]

        mask_number = iter(range(len(predicted_tokens)))
        text_to_display = "".join([tokens[i] if i+1 not in mask_positions
                                             else '<' + str(next(mask_number)) + '>'
                                             for i in range(len(tokens))]) + 2*'\n'

        for i in sorted(predicted_tokens, key=lambda x: x['position']):
            text_to_display += (f"{'<':>3}{mask_positions.index(i['position']):>2}>|{i['original_token_id']:15}|{i['prediction_ids']}\n")

        return text_to_display

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

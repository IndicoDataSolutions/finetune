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
                "mlm_weights": tf.float32,
                "mlm_ids": tf.int32,
                "mlm_positions": tf.int32
            }
        shapes = {
                "tokens": TS([None]),
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
            mlm_weights = np.ones_like(mlm_ids)

            
            out.token_ids[:, 0][mlm_mask & (mask_type == 'mask')] = self.text_encoder.mask_token
            #print("out.token_ids:", out.token_ids)
            out.token_ids[:, 0][mlm_mask & (mask_type == 'random')] = random_tokens[mlm_mask & (mask_type == 'random')]

            feats = {
                "tokens": out.token_ids, 
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

    defaults = dict(low_memory_mode=True, batch_size=48, xla=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not issubclass(self.config.base_model, (BERT, RoBERTa)):
            raise FinetuneError("MLM training is currently only supported for BERT and RoBERTa base models.")

    def _get_input_pipeline(self):
        return MaskedLanguageModelPipeline(self.config)


    def display_text_(self, input_text, k=5, context=None, **kwargs):

        def dataset_encoded():
            while not dataset_encoded.finished:
                yield {"tokens": arr_encoded.token_ids, "mask": arr_encoded.mask}

        dataset_encoded.finished = False

        def get_input_fn():
            types, shapes = self.input_pipeline.feed_shape_type_def()
            tf_dataset = Dataset.from_generator(dataset_encoded, types[0], shapes[0])
            return tf_dataset.batch(1)

        encoded = self.input_pipeline.text_encoder._encode([input_text])
        token_ids = encoded.token_ids[0]
        encoded = EncodedOutput(token_ids=token_ids)
        estimator, hooks = self.get_estimator(force_build_lm=True)
        predict = estimator.predict(input_fn=get_input_fn,
                                    predict_keys=[PredictMode.GENERATE_TEXT,],
                                                 # PredictMode.MLM_IDS,
                                                 # PredictMode.MLM_POSITIONS],
                                                  hooks=hooks)
        arr_encoded = self.input_pipeline._array_format(encoded)
        prediction_information = list(predict)

        return prediction_information


    def display_text(self, input_text, k=5, context=None, **kwargs):
        prediction_info = self._inference([input_text],
                                          predict_keys=[PredictMode.GENERATE_TEXT,
                                                        PredictMode.MLM_IDS,
                                                        PredictMode.MLM_POSITIONS],
                                          context=context,
                                          force_build_lm=True,
                                          **kwargs)
        #print("raw_preds:", raw_preds)
        #print(np.asarray(raw_preds).shape)
        #return self.input_pipeline.text_encoder.decode(
        #    np.asarray(raw_preds)
        #)

        #text_to_display = list(self.input_pipeline.text_to_tokens_mask(input_text))[0]['tokens'][:,0]
        #text_to_display = "".join([self.input_pipeline.text_encoder.decoder[token_id] for token_id in text_to_display])
        #text_to_display = bytearray([self.input_pipeline.text_encoder.byte_decoder[c] for c in text_to_display]).decode("utf-8", errors=self.input_pipeline.text_encoder.errors)
        #print('Text encoded with _encode:', self.input_pipeline.text_encoder._encode([input_text]))
        #text = "".join([self.decoder[token_id] for token_id in token_ids])
        #text = bytearray([self.byte_decoder[c] for c in text]).decode(
        #    "utf-8", errors=self.errors
        #)
        #print(prediction_info)
        predicted_tokens = [{'prediction_ids': [self.input_pipeline.text_encoder.decode([i]) for i in pred['GEN_TEXT'][-k:][::-1]],
                             'original_token_id': self.input_pipeline.text_encoder.decode([pred['mlm_ids']]),
                             'position': pred['mlm_positions']} for pred in prediction_info]
        #import ipdb; ipdb.set_trace()
        mask_positions = [i['position'] for i in predicted_tokens]

        tokens = self.input_pipeline.text_encoder._encode([input_text]).tokens[0]
        #print("tokens:",tokens)

        mask_number = iter(range(len(predicted_tokens)))
        text_to_display = "".join([tokens[i] if i+1 not in mask_positions
                                             else '<' + str(next(mask_number)) + '>'
                                             for i in range(len(tokens))]) + 2*'\n'
        #print(text_to_display)

        for i in sorted(predicted_tokens, key=lambda x: x['position']):
            text_to_display += (f"<{mask_positions.index(i['position'])}>\t|{i['original_token_id']}|\t{i['prediction_ids']}\n")
        #print(text_to_display)

        return text_to_display
        #return [self.input_pipeline.text_encoder.decoder[id] for mask in raw_preds for id in mask]
    def generate_text(self, input_text, context=None, **kwargs):

        #def dataset_encoded():
        #    while not dataset_encoded.finished:
        #        yield {"tokens": arr_encoded.token_ids, "mask": arr_encoded.mask}
        #def get_input_fn():
        #    types, shapes = self.input_pipeline.feed_shape_type_def()
        #    tf_dataset = Dataset.from_generator(dataset_encoded, types[0], shapes[0])
        #    return tf_dataset.batch(1)

        #encoded = self.input_pipeline.text_encoder._encode([input_text])

        #encoded = self.input_pipeline.text_to_tokens_mask([input_text])
        #print(encoded)
        #if encoded.token_ids == []: #and not use_extra_toks:
        #    raise ValueError(
        #        "If you are not using the extra tokens, you must provide some non-empty seed text"
        #    )
        #start = []
        #token_ids = start
        #if encoded.token_ids is not None and len(encoded.token_ids):
        #    token_ids += encoded.token_ids[0]
        #encoded = EncodedOutput(token_ids=token_ids)
        #input_fn=get_input_fn
        #raw_preds = self._inference([input_text], predict_keys=[PredictMode.GENERATE_TEXT], context=context, force_build_lm=True, **kwargs)
        raw_preds = self._inference([input_text], predict_keys=[PredictMode.GENERATE_TEXT, PredictMode.MLM_IDS], context=context, force_build_lm=True, **kwargs)
        #print("raw_preds:", raw_preds)
        #print(np.asarray(raw_preds).shape)
        return self.input_pipeline.text_encoder.decode(
            np.asarray(raw_preds)
        )

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

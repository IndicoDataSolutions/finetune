
import tensorflow as tf

from finetune.base import BaseModel
from finetune.encoding.target_encoders import Seq2SeqLabelEncoder
from finetune.input_pipeline import BasePipeline
from finetune.util.shapes import shape_list
from finetune.nn.target_blocks import language_model
from finetune.model import PredictMode
from finetune.util.beam_search import beam_search


class S2SPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_dim = 0
        self.label_encoder = self._target_encoder()

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        return (
            (
                {
                    "tokens": tf.int32,
                },
                (tf.int32, tf.int32)
            ),
            (
                {
                    "tokens": TS([None]),
                },
                (TS([None]), TS([]))
            )
        )

    def _target_encoder(self):
        return Seq2SeqLabelEncoder(self.text_encoder, self.config.max_length)

def normalize_embeds(embeds):
    return embeds * (tf.cast(tf.shape(embeds)[-1], dtype=tf.float32) ** -0.5)

class HFS2S(BaseModel):

    def _get_input_pipeline(self):
        pipeline = S2SPipeline(self.config)
        return pipeline

    @staticmethod
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, label_encoder=None, **kwargs):
        text_encoder = label_encoder.encoder
        hf_decoder = featurizer_state["decoder"]
        encoder_decoder_mask = tf.sequence_mask(featurizer_state["lengths"], dtype=tf.float32)
        if targets is not None:
            targets, lengths = targets
            padding_mask = tf.sequence_mask(lengths - 1, dtype=tf.float32)
            embeds = hf_decoder(
                (
                    targets[:, :-1], #decoder_input_ids, # cuts final token off internally?
                    padding_mask, #decoder_attention_mask,
                    featurizer_state["sequence_features"], #hidden_states,
                    encoder_decoder_mask, #encoder_attention_mask,
                    None, #decoder_inputs_embeds,
                    None, #head_mask,
                    None, #decoder_past_key_value_states,
                    False, #use_cache,
                    None, #output_attentions,
                    None, #output_hidden_states,
                ),
                training=train,
            )[0]
            logits = featurizer_state["embed_weights"](normalize_embeds(embeds), mode="linear")
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=targets[:, 1:], 
            )
            padding_mask = tf.compat.v1.Print(padding_mask, [padding_mask], summarize=100)
            loss = loss * padding_mask
            loss = tf.reduce_sum(loss)
            return {
                "logits": logits,
                "losses": loss
            }

        else:
            def symbols_to_logits_fn(input_symbols, i, state): #[batch_size, decoded_ids] to [batch_size, vocab_size]
                with tf.compat.v1.variable_scope("model"):
                    with tf.compat.v1.variable_scope("target"):
                        input_symbols = tf.compat.v1.Print(input_symbols, [input_symbols])
                        embeds = hf_decoder(
                            (
                                input_symbols, #decoder_input_ids,
                                None, #decoder_attention_mask, # Seems like it does this automagically because these values are unpadded
                                state["encoder_output"], #hidden_states,
                                state["encoder_decoder_mask"], #encoder_attention_mask,
                                None, #decoder_inputs_embeds,
                                None, #head_mask,
                                None, #decoder_past_key_value_states,
                                False, #use_cache,
                                None, #output_attentions,
                                None, #output_hidden_states,
                            ),
                            training=False,
                        )[0]
                        logits = featurizer_state["embed_weights"](normalize_embeds(embeds[:, -1]), mode="linear")
                        logits_shape = tf.shape(logits)
                        #logits = tf.concat((tf.zeros(shape=[logits_shape[0], 1], dtype=tf.float32), logits[:, 1:]), 1)
                        return (logits, state)

            initial_ids = tf.tile(tf.constant([text_encoder.start_token], dtype=tf.int32), [tf.shape(featurizer_state["sequence_features"])[0]])

            beams, probs, _ = beam_search(
                symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=initial_ids,
                beam_size=config.beam_size,
                decode_length=config.max_length,
                vocab_size=featurizer_state["embed_weights"].vocab_size,
                alpha=config.beam_search_alpha,
                states={"encoder_output": featurizer_state["sequence_features"], "encoder_decoder_mask": encoder_decoder_mask}, # TODO: Use states to enable cache in t5
                eos_id=text_encoder.end_token,
                stop_early=True,
                use_top_k_with_unique=True,
            )

            best_beams_i = tf.argmax(probs, -1, output_type=tf.int32)
            best_beam = tf.gather_nd(beams, tf.stack([tf.range(tf.shape(beams)[0]), best_beams_i], -1))
            return {
                "logits": best_beam,
                "losses": -1.0
            }

    def _predict_op(self, logits):
        return logits

    def _predict_proba_op(self, logits):
        return logits

    def _predict(self, zipped_data, **kwargs):
        preds = self._inference(zipped_data, predict_keys=[PredictMode.NORMAL],  **kwargs)
        print(preds)
        return self.input_pipeline.label_encoder.inverse_transform(preds)

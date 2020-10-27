import functools

import numpy as np
import tensorflow as tf

from finetune.base import BaseModel
from finetune.encoding.target_encoders import Seq2SeqLabelEncoder
from finetune.input_pipeline import BasePipeline
from finetune.model import PredictMode
from finetune.util.beam_search import beam_search, get_state_shape_invariants
from tensorflow.python.util import nest
import itertools
import sys

def py_bracket_constraint(history, bracket_idx_pairs, eos_token, vocab_size):
    vocab_mask = np.ones([vocab_size], dtype=np.float32)
    stack = []
    opening_brackets = [b[0] for b in bracket_idx_pairs]
    closing_brackets = [b[1] for b in bracket_idx_pairs]
    for tok in history.tolist():
        print(stack, tok, opening_brackets, closing_brackets)
        if tok in opening_brackets:
            stack.append(opening_brackets.index(tok)) # idx in bracket_idx_pairs
        if tok in closing_brackets:
            print("Cannor enforce for: ", history)
            assert closing_brackets.index(tok) == stack.pop()
    if stack:
        stack_head = stack[-1]
        vocab_mask[eos_token] = 0 # do not allow the sequence to end here.
    else:
        stack_head = None # Do not allow any closing brackets

    for i, cb in enumerate(closing_brackets):
        if stack_head == i:
            continue # allow the current stack head to be closed
        vocab_mask[cb] = 0 # mask all other bracket closes
    return vocab_mask


class BracketConstrainedPredictions:
    def __init__(self, bracket_idx_pairs, eos_token, vocab_size):
        self.bracket_idx_pairs = bracket_idx_pairs
        self.eos_token = eos_token
        self.vocab_size = vocab_size

    def mask_for_batched_history(self, history):
        def tf_bracket_constraint_single(single_history):
            return tf.numpy_function(
                functools.partial(
                    py_bracket_constraint,
                    bracket_idx_pairs=self.bracket_idx_pairs,
                    eos_token=self.eos_token,
                    vocab_size=self.vocab_size
                ), [single_history], tf.float32, name=None
            )
        return tf.map_fn(tf_bracket_constraint_single, history, parallel_iterations=100, back_prop=False, dtype=tf.float32)


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
        return Seq2SeqLabelEncoder(self, self.config.s2s_decoder_max_length)

def normalize_embeds(embeds):
    return embeds * (tf.cast(tf.shape(embeds)[-1], dtype=tf.float32) ** -0.5)


def label_smooth(targets, pad_mask, smoothing, smooth_mean_targets):
    # targets: (batch, seq, n_vocab)
    if smooth_mean_targets:
        # Only allocate extra proba mass to token ids in output seq
        token_count = tf.reduce_sum(targets * tf.expand_dims(pad_mask, axis=-1), axis=1, keepdims=True)
        seq_len = tf.reduce_sum(token_count, axis=-1, keepdims=True) 
        targets = targets * (1. - smoothing) + smoothing * token_count / seq_len
    else:
        # Allocate extra proba mass over all tokens
        targets = targets * (1. - smoothing) + smoothing / targets.shape[-1]

    return targets

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
            logits = featurizer_state["embedding"](normalize_embeds(embeds), mode="linear")
            final_targets = tf.one_hot(targets[:, 1:], depth=logits.shape[-1])

            if config.s2s_label_smoothing:
                final_targets = label_smooth(
                    targets=final_targets, 
                    pad_mask=padding_mask, 
                    smoothing=config.s2s_label_smoothing, 
                    smooth_mean_targets=config.s2s_smoothing_mean_targets
                )

            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=final_targets
            )
            loss = loss * padding_mask
            loss = tf.reduce_sum(loss)
            return {
                "logits": logits,
                "losses": loss
            }

        else:
            if config.delim_tokens:
                # Build list of tokens allowed for all examples
                encoded_delim = text_encoder._encode(config.delim_tokens)[0]
                encoded_delim = list(set(flatten(encoded_delim)))
                encoded_delim.append(text_encoder.end_token)
                # Tile encoded delims to be shape [batch_size, num delim]
                encoded_delim = tf.constant(encoded_delim)[None, :]
                encoded_delim = tf.tile(encoded_delim,
                                        (tf.shape(featurizer_state["inputs"])[0], 1))
                # Add delim tokens to all examples
                constraint = tf.concat((featurizer_state["inputs"], encoded_delim), axis=-1)

            if config.bracket_constraints:
                bracket_idx_pairs = []
                for ob, cb in config.bracket_constraints:
                    ob_idx = text_encoder._encode([ob])
                    cb_idx =  text_encoder._encode([cb])
                    bracket_idx_pairs.append((ob_idx.token_ids[0][-1], cb_idx.token_ids[0][-1]))
                bracket_constrainer = BracketConstrainedPredictions(
                    bracket_idx_pairs,
                    text_encoder.end_token,
                    featurizer_state["embedding"].vocab_size
                )

            def symbols_to_logits_fn(input_symbols, i, state, first=False): #[batch_size, decoded_ids] to [batch_size, vocab_size]
                embeds, present_state = hf_decoder(
                    (
                        input_symbols[:, -1][:, None], #decoder_input_ids,
                        None, #decoder_attention_mask, # Seems like it does this automagically because these values are unpadded
                        state["encoder_output"], #hidden_states,
                        state["encoder_decoder_mask"], #encoder_attention_mask,
                        None, #decoder_inputs_embeds,
                        None, #head_mask,
                        None if first else state["past_states"], #decoder_past_key_value_states,
                        True, #use_cache,
                        None, #output_attentions,
                        None, #output_hidden_states,
                    ),
                    training=False,
                )
                logits = featurizer_state["embedding"](normalize_embeds(embeds[:, -1]), mode="linear")
                logits_shape = tf.shape(logits)

                if config.delim_tokens or config.bracket_constraints:
                    if config.delim_tokens:
                        batch_indices = tf.range(tf.shape(state["constraint"])[0])[:, None] + \
                                tf.zeros_like(state["constraint"])
                        full_indices = tf.reshape(tf.stack([batch_indices, state["constraint"]], axis=2), [-1,2])
                        mask = tf.scatter_nd(full_indices,
                                            tf.ones((tf.shape(full_indices)[0],)),
                                            shape=tf.shape(logits))
                        # Normalize for the way scatter_nd handles duplicates
                        mask = tf.math.divide_no_nan(mask, mask)
                    else:
                        mask = 1.0

                    if config.bracket_constraints:
                        mask = mask * bracket_constrainer.mask_for_batched_history(input_symbols)
                    logits = logits * mask + ((1 - mask) * -1e5)
                
                state["past_states"] = present_state
                return (logits, state)

            initial_ids = tf.tile(tf.constant([text_encoder.start_token], dtype=tf.int32), [tf.shape(featurizer_state["sequence_features"])[0]])
            batch_size = tf.shape(initial_ids)[0]
            past_states = tuple(
                (
                    tf.zeros((batch_size, hf_decoder.config.num_heads, 0, hf_decoder.config.d_kv)),
                    tf.zeros((batch_size, hf_decoder.config.num_heads, 0, hf_decoder.config.d_kv)),
                    tf.zeros((batch_size, hf_decoder.config.num_heads, 0, hf_decoder.config.d_kv)),
                    tf.zeros((batch_size, hf_decoder.config.num_heads, 0, hf_decoder.config.d_kv)),
                )
            for _ in range(len(hf_decoder.block)))

            states = {
                "encoder_output": featurizer_state["sequence_features"],
                "encoder_decoder_mask": encoder_decoder_mask,
                "past_states": past_states,
            }
            if config.delim_tokens:
                states["constraint"] = constraint
            beams, probs, _ = beam_search(
                symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=initial_ids,
                beam_size=config.beam_size,
                decode_length=config.s2s_decoder_max_length,
                vocab_size=featurizer_state["embedding"].vocab_size,
                alpha=config.beam_search_alpha,
                states=states,
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
        return self.input_pipeline.label_encoder.inverse_transform(preds)

def flatten(items, seqtypes=(list, tuple)):
    # https://stackoverflow.com/a/10824086
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items

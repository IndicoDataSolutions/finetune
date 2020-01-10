import tensorflow as tf
from finetune.base import BaseModel, PredictMode
from finetune.encoding.target_encoders import Seq2SeqLabelEncoder
from finetune.input_pipeline import BasePipeline
from finetune.util.shapes import shape_list
from finetune.nn.target_blocks import language_model
from finetune.util.beam_search import beam_search
from finetune.base_models.oscar.featurizer import featurizer as gpc_featurizer
from finetune.errors import FinetuneError


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
                    "mask": tf.float32
                },
                tf.int32
            ),
            (
                {
                    "tokens": TS([None, 2]),
                    "mask": TS([None])
                },
                TS([None, 2])
            )
        )

    def _target_encoder(self):
        return Seq2SeqLabelEncoder(self.text_encoder, self.config.max_length)


class S2S(BaseModel):
    """ 
    Classifies a single document into 1 of N categories.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def __init__(self, **kwargs):
        super().__init__(target_model_init_from_base_model=True, **kwargs)
        if self.config.base_model.featurizer not in [gpc_featurizer]:
            raise FinetuneError(
                "{} base model is not supported for {}".format(self.config.base_model.__name__, self.__name__)
            )

    def _get_input_pipeline(self):
        pipeline = S2SPipeline(self.config)
        return pipeline

    def featurize(self, X, **kwargs):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return super().featurize(X, **kwargs)


    def predict(self, X, **kwargs):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return super().predict_proba(X, **kwargs)

    def finetune(self, X, Y=None, batch_size=None, **kwargs):
        """
        :param X: list or array of text.
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=Y, batch_size=batch_size, **kwargs)

    @staticmethod
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, label_encoder=None, **kwargs):
        encoder = label_encoder.encoder
        if targets is not None:
            target_feat_state = config.base_model.get_featurizer(
                targets,
                encoder=encoder,
                config=config,
                train=train,
                encoder_state=featurizer_state
            )
            embed_weights = target_feat_state["embed_weights"]
            return language_model(
                X=targets,
                M=tf.sequence_mask(target_feat_state["eos_idx"] + 1, maxlen=shape_list(embed_weights)[1], dtype=tf.float32), # +1 because we want to predict the clf token as an eos token
                embed_weights=embed_weights,
                config=config,
                reuse=reuse, train=train,
                hidden=target_feat_state["sequence_features"]
            )
        else:
            # returns (decoded beams [batch_size, beam_size, decode_length]
            #          decoding probabilities [batch_size, beam_size])
            embed_weights = kwargs.get("embed_weights") if "embed_weights" in kwargs else featurizer_state.pop("embed_weights")
            
            def symbols_to_logits_fn(input_symbols, i, state):
                # [batch_size, decoded_ids] to [batch_size, vocab_size]
                leng = shape_list(input_symbols)[1]
                pos_embed = encoder.vocab_size + tf.range(leng)
                pos_embed = tf.tile([pos_embed], [shape_list(input_symbols)[0], 1])
                inp = tf.pad(tf.stack([input_symbols, pos_embed], -1), [[0,0], [0, 1], [0, 0]] )
                target_feat_state = config.base_model.get_featurizer(
                    inp,
                    encoder=encoder,
                    config=config,
                    train=train,
                    encoder_state=({**state["featurizer_state"], "embed_weights":embed_weights} if state else None)
                )
                output_state = language_model(
                    X=inp,
                    M=tf.sequence_mask(target_feat_state["eos_idx"] + 1, maxlen=leng + 1, dtype=tf.float32), # +1 because we want to predict the clf token as an eos token
                    embed_weights=embed_weights[:encoder.vocab_size, :],
                    config=config,
                    reuse=reuse, train=train,
                    hidden=target_feat_state["sequence_features"] # deal with state
                )
                if state is None:
                    return output_state["logits"][:, i, :]
                else:
                    return output_state["logits"][:, i, :], state
            start_tokens = kwargs.get("start_tokens", tf.constant([encoder.start_token for _ in range(config.batch_size)], dtype=tf.int32))
            beams, probs, _ = beam_search(
                symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=start_tokens,
                beam_size=config.beam_size,
                decode_length=tf.minimum(config.max_length, tf.shape(start_tokens)[1] + config.seq_decode_len),
                vocab_size=encoder.vocab_size,
                alpha=config.beam_search_alpha,
                states={"featurizer_state": featurizer_state} if featurizer_state is not None else {},
                eos_id=encoder.end_token,
                stop_early=True,
                use_top_k_with_unique=True,
                temperature=config.sample_temp,
                sample_from_top=config.decoder_sample_from
            )

            best_beams_i = tf.argmax(probs, -1, output_type=tf.int32)
            best_beam = tf.gather_nd(beams, tf.stack([tf.range(tf.shape(beams)[0]), best_beams_i], -1))
            return {
                "logits": best_beam,
                "losses": -1.0
            }

    def _predict_op(self, logits, **kwargs):
        return logits

    def _predict_proba_op(self, logits, **kwargs):
        return logits


class LMPred(S2S):
    @staticmethod
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, label_encoder=None, **kwargs):
        start_tokens = featurizer_state["encoded_input"]
        return S2S._target_model(
            config, None, targets, n_outputs, train=train, reuse=reuse, label_encoder=label_encoder,
            embed_weights=featurizer_state["embed_weights"], start_tokens=start_tokens
        )

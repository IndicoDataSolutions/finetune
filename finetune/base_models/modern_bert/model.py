import os
import tensorflow as tf
from finetune.base_models.modern_bert.modelling import ModernBert, ModernBertConfig
from finetune.base_models.modern_bert.encoding import ModernBertEncoder
from finetune.base_models import SourceModel


def featurizer(
        X, encoder, config, train=False, reuse=None, lengths=None, **kwargs
    ):
    initial_shape = tf.shape(input=X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))
    X.set_shape([None, None])
    delimiters = tf.cast(tf.equal(X, encoder.delimiter_token), tf.int32)

    seq_length = tf.shape(input=delimiters)[1]
    mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)
    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        # TODO: plumb in the config to the finetune config.
        model = ModernBert(config=ModernBertConfig())
        embedding = model.embeddings
        sequence_out = model(input_ids=X, attention_mask=mask, training=train, seq_len=seq_length)
        pooled_out = sequence_out[:, 0, :]
        pooled_out.set_shape([None, config.n_embed])
        n_embed = pooled_out.shape[-1]

        features = tf.reshape(
            pooled_out,
            shape=tf.concat((initial_shape[:-1], [n_embed]), 0),
        )
        sequence_features = tf.reshape(
            sequence_out,
            shape=tf.concat((initial_shape, [n_embed]), 0),
        )

        output_state = {
            "embedding": embedding,
            "features": features,
            "sequence_features": sequence_features,
            "lengths": lengths,
            "inputs": X,
        }

        return output_state


class ModernBertModel(SourceModel):
    encoder = ModernBertEncoder
    featurizer = featurizer
    max_length = 2048

    settings = {
        "base_model_path": os.path.join("modern_bert", "modern_bert.jl"),
        "n_layer": 22,
        "train_embeddings": True,
        "num_layers_trained": 22,
        "n_embed": 768,
        "max_length": max_length,
        "include_bos_eos": True,
    }
    required_files = []


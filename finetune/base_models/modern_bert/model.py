import os
from urllib.parse import urljoin
import tensorflow as tf
from finetune.base_models.modern_bert.modelling import ModernBert
from finetune.base_models.modern_bert.encoding import ModernBertEncoder
from finetune.base_models import SourceModel
from finetune.util.download import MODERN_BERT_BASE_URL, FINETUNE_BASE_FOLDER

def featurizer(X, encoder, config, train=False, reuse=None, lengths=None, **kwargs):
    initial_shape = tf.shape(input=X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))
    X.set_shape([None, None])
    delimiters = tf.cast(tf.equal(X, encoder.delimiter_token), tf.int32)

    seq_length = tf.shape(input=delimiters)[1]
    mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)
    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        model = ModernBert(
            config=config,
            vocab_size=encoder.vocab_size,
        )
        embedding = model.embeddings
        sequence_out = model(
            input_ids=X, attention_mask=mask, training=train, seq_len=seq_length
        )
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
    max_length = 512
    is_bidirectional = True


    settings = {
        "base_model_path": os.path.join("modern_bert", "modern_bert.jl"),
        "n_layer": 22,
        "train_embeddings": True,
        "num_layers_trained": 22,
        "n_embed": 768,
        "max_length": max_length,
        "include_bos_eos": True,
        "n_heads": 12,
        "batch_size": 8,
        "bert_intermediate_size": 1152,
        "lr": 3e-4, # previously 1.5e-4
        "n_epochs": 12, # previously 6
        "max_grad_norm": 3.0,
        "l2_reg": 5e-4,
        "lr_warmup": 0.25, # this seems high but is what the sweep ended up with.
        "lr_schedule": "warmup_linear",
        "low_memory_mode": True,
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "modern_bert", filename),
            "url": urljoin(MODERN_BERT_BASE_URL, filename),
        }
        for filename in ["modern_bert.jl", "tokenizer.json"]
    ]

    @classmethod
    def get_optimal_params(cls, config):
        base_n_epochs = config.base_model.settings["n_epochs"]
        base_learning_rate = config.base_model.settings["lr"]
        if config.optimize_for.lower() in ["accuracy", "accuracy_fp16"]:
            overrides = {
                "max_length": 2048,
                "n_epochs": base_n_epochs,
                "batch_size": 8,
                "chunk_context": None,
                "predict_batch_size": 16,
                "mixed_precision": True,
                "float_16_predict": True,
                "lr": base_learning_rate,
            }

        elif config.optimize_for.lower() in ["predict_speed", "predict_speed_fp16"]:
            overrides = {
                "max_length": 512,
                "n_epochs": base_n_epochs,
                "batch_size": 24,
                "chunk_context": 16,
                "predict_batch_size": 32, # We can fit a lot more in memory but 32 seems most efficient.
                "mixed_precision": True,
                "float_16_predict": True,
                "lr": base_learning_rate,
            }
        else:
            raise ValueError(
                "Cannot optimise hyperparams for {}, must be either 'speed', 'predict_speed' or 'accuracy'".format(
                    config.optimize_for
                )
            )
        return overrides


class ModernBertLargeModel(SourceModel):
    is_bidirectional = True
    encoder = ModernBertEncoder
    featurizer = featurizer
    max_length = 512

    settings = {
        "base_model_path": os.path.join("modern_bert", "modern_bert_large.jl"),
        "n_layer": 28,
        "train_embeddings": True,
        "num_layers_trained": 28,
        "n_embed": 1024,
        "max_length": max_length,
        "include_bos_eos": True,
        "n_heads": 16,
        "bert_intermediate_size": 2624,
        "batch_size": 4,
 #       "accum_steps": 16,
        "lr": 5e-4, # previously 1.5e-4
        "n_epochs": 8, # previously 6
        "max_grad_norm": 3.0,
        "l2_reg": 5e-4,
        "lr_warmup": 0.25, # this seems high but is what the sweep ended up with.
        "lr_schedule": "warmup_linear",
        "low_memory_mode": True,
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "modern_bert", filename),
            "url": urljoin(MODERN_BERT_BASE_URL, filename),
        }
        for filename in ["modern_bert_large.jl", "tokenizer.json"]
    ]
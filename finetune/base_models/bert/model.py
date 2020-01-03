import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.bert.encoder import (
    BERTEncoder,
    BERTEncoderMultuilingal,
    BERTEncoderLarge,
    DistilBERTEncoder
)

from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder, RoBERTaEncoderV2
from finetune.base_models.bert.featurizer import bert_featurizer, bert_featurizer_with_pos
from finetune.base_models.gpt2 import encoder as gpt2_encoder
from finetune.util.download import BERT_BASE_URL, GPT2_BASE_URL, ROBERTA_BASE_URL, FINETUNE_BASE_FOLDER


class _BaseBert(SourceModel):
    is_bidirectional = True

    @classmethod
    def get_optimal_params(cls, config):
        base_max_length = config.base_model.settings.get("max_length", 512)
        if config.optimize_for.lower() == "speed":
            overrides = {
                "max_length": 128 if config.chunk_long_sequences else base_max_length,
                "n_epochs": 5,
                "batch_size": 4,
                "chunk_context": 16,
                "predict_batch_size": 48,
            }

        elif config.optimize_for.lower() == "accuracy":
            overrides = {
                "max_length": base_max_length,
                "n_epochs": 8,
                "batch_size": 2,
                "chunk_context": None,
                "predict_batch_size": 20,
            }

        elif config.optimize_for.lower() == "predict_speed":
            overrides = {
                "max_length": 128 if config.chunk_long_sequences else base_max_length,
                "n_epochs": 8,
                "batch_size": 2,
                "chunk_context": 16,
                "predict_batch_size": 48,
            }
        else:
            raise ValueError(
                "Cannot optimise hyperparams for {}, must be either 'speed', 'predict_speed' or 'accuracy'".format(
                    config.optimize_for
                )
            )
        return overrides


class BERTModelCased(_BaseBert):
    encoder = BERTEncoder
    featurizer = bert_featurizer
    settings = {
        "lm_type": "mlm",
        "n_embed": 768,
        "n_epochs": 8,
        "n_heads": 12,
        "n_layer": 12,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_intermediate_size": 3072,
        "base_model_path": os.path.join("bert", "bert_small_cased-v2.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_small_cased-v2.jl", "vocab.txt"]
    ]


class BERTModelLargeCased(_BaseBert):
    encoder = BERTEncoderLarge
    featurizer = bert_featurizer
    settings = {
        "lm_type": "mlm",
        "n_embed": 1024,
        "n_epochs": 8,
        "n_heads": 16,
        "n_layer": 24,
        "num_layers_trained": 24,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_intermediate_size": 4096,
        "base_model_path": os.path.join("bert", "bert_large_cased-v2.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_large_cased-v2.jl", "vocab_large.txt"]
    ]


class BERTModelLargeWWMCased(_BaseBert):
    encoder = BERTEncoderLarge
    featurizer = bert_featurizer
    settings = {
        "lm_type": "mlm",
        "n_embed": 1024,
        "n_epochs": 8,
        "n_heads": 16,
        "n_layer": 24,
        "num_layers_trained": 24,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_intermediate_size": 4096,
        "base_model_path": os.path.join("bert", "bert_wwm_large_cased-v2.jl"),
        "max_length": 1024,
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_wwm_large_cased-v2.jl", "vocab_large.txt"]
    ]


class BERTModelMultilingualCased(_BaseBert):
    encoder = BERTEncoderMultuilingal
    featurizer = bert_featurizer
    settings = {
        "lm_type": "mlm",
        "n_embed": 768,
        "n_epochs": 8,
        "n_heads": 16,
        "n_layer": 12,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_intermediate_size": 3072,
        "base_model_path": os.path.join("bert", "bert_small_multi_cased-v2.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_small_multi_cased-v2.jl", "vocab_multi.txt"]
    ]


class RoBERTa(_BaseBert):
    encoder = RoBERTaEncoderV2
    featurizer = bert_featurizer
    settings = {
        "lm_type": "mlm",
        "n_embed": 768,
        "n_epochs": 8,
        "n_heads": 12,
        "n_layer": 12,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.1,
        "epsilon": 1e-8,
        "bert_intermediate_size": 3072,
        "bert_use_pooler": False,
        "base_model_path": os.path.join("bert", "roberta-model-sm-v2.jl"),
    }
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta-model-sm-v2.jl"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta-model-sm-v2.jl"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "dict.txt"),
            "url": urljoin(ROBERTA_BASE_URL, "dict.txt"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]

    @classmethod
    def get_encoder(cls, config=None, **kwargs):
        roberta_filename = config.base_model_path.rpartition('/')[-1]
        missing_mask_token = roberta_filename in ('roberta-model-sm.jl', 'roberta-model-lg.jl')
        if config is not None and not missing_mask_token:
            return cls.encoder(**kwargs)
        else:
            return RoBERTaEncoder(**kwargs)


class RoBERTaPos(RoBERTa):
    positional_info = True
    # TODO: ideally we would pull out the positional logic so that
    # it can be easily incorporated throughout all the SourceModels
    featurizer = bert_featurizer_with_pos
    @classmethod
    def get_encoder(cls, config=None, **kwargs):
        if config is not None and not config.base_model_path.endswith('-v2'):
            return cls.encoder(**kwargs)
        else:
            return RoBERTaEncoder(**kwargs)

class RoBERTaLarge(RoBERTa):
    encoder = RoBERTaEncoderV2
    featurizer = bert_featurizer
    settings = {
        "lm_type": "mlm",
        "n_embed": 1024,
        "n_epochs": 8,
        "n_heads": 16,
        "n_layer": 24,
        "num_layers_trained": 24,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.1,
        "epsilon": 1e-8,
        "bert_intermediate_size": 4096,
        "bert_use_pooler": False,
        "base_model_path": os.path.join("bert", "roberta-model-lg-v2.jl"),
    }
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta-model-lg-v2.jl"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta-model-lg-v2.jl"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "dict.txt"),
            "url": urljoin(ROBERTA_BASE_URL, "dict.txt"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]

   
ZuckerBERT = RoBERTa


class DistilBERT(_BaseBert):
    encoder = DistilBERTEncoder
    featurizer = bert_featurizer
    settings = {
        "n_embed": 768,
        "n_epochs": 8,
        "n_heads": 12,
        "n_layer": 6,
        "num_layers_trained": 6,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_use_pooler": False,
        "bert_use_type_embed": False,
        "bert_intermediate_size": 3072,
        "base_model_path": os.path.join("bert", "distillbert.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["distillbert.jl", "distillbert_vocab.txt"]
    ]


class DistilRoBERTa(_BaseBert):
    encoder = RoBERTaEncoder
    featurizer = bert_featurizer
    settings = {
        "n_embed": 768,
        "n_epochs": 8,
        "n_heads": 12,
        "n_layer": 6,
        "num_layers_trained": 6,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_use_pooler": False,
        "bert_use_type_embed": False,
        "bert_intermediate_size": 3072,
        "base_model_path": os.path.join("bert", "distilroberta.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", 'distilroberta.jl'),
            "url": urljoin(BERT_BASE_URL, 'distilroberta.jl'),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "dict.txt"),
            "url": urljoin(ROBERTA_BASE_URL, "dict.txt"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "gpt2", "vocab.bpe"),
            "url": urljoin(GPT2_BASE_URL, "vocab.bpe"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "gpt2", "encoder.json"),
            "url": urljoin(GPT2_BASE_URL, "encoder.json"),
        }
    ]

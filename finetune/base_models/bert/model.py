import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.bert.encoder import (
    BERTEncoder,
    BERTEncoderMultuilingal,
    BERTEncoderLarge,
    DistilBERTEncoder,
    LayoutLMEncoder,
)

from finetune.base_models.bert.roberta_encoder import (
    RoBERTaEncoder,
    RoBERTaEncoderV2,
    RoBERTaEncoderXDoc,
)
from finetune.base_models.bert.featurizer import (
    bert_featurizer,
    layoutlm_featurizer,
    xdoc_featurizer,
    table_roberta_featurizer_twinbert,
)

from finetune.util.context_utils import get_context_layoutlm, get_context_doc_rep
from finetune.util.download import (
    BERT_BASE_URL,
    GPT2_BASE_URL,
    ROBERTA_BASE_URL,
    LAYOUTLM_BASE_URL,
    FINETUNE_BASE_FOLDER,
)
from finetune.util.featurizer_fusion import fused_featurizer

BERT_BASE_PARAMS = {
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
}

BERT_LARGE_PARAMS = {
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
    "max_length": 512,
}

DISTIL_BERT_PARAMS = {
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
}


class _BaseBert(SourceModel):
    is_bidirectional = True
    is_roberta = False

    @classmethod
    def get_optimal_params(cls, config):
        base_max_length = config.base_model.settings.get("max_length", 512)
        base_n_epochs = config.base_model.settings.get("n_epochs", 8)
        base_batch_size = config.base_model.settings.get("batch_size", 2)
        base_learning_rate = config.base_model.settings.get("lr", 1e-5)
        if config.optimize_for.lower() == "speed":
            overrides = {
                "max_length": 128 if config.chunk_long_sequences else base_max_length,
                "n_epochs": 5,
                "batch_size": 4,
                "chunk_context": 16,
                "predict_batch_size": 256 if config.float_16_predict else 48,
                "mixed_precision": False,
                "float_16_predict": False,
                "lr": base_learning_rate,
            }

        elif config.optimize_for.lower() == "accuracy":
            overrides = {
                "max_length": base_max_length,
                "n_epochs": base_n_epochs,
                "batch_size": base_batch_size,
                "chunk_context": None,
                "predict_batch_size": 20,
                "mixed_precision": False,
                "float_16_predict": False,
                "lr": base_learning_rate,
            }

        elif config.optimize_for.lower() == "predict_speed":
            overrides = {
                "max_length": 128 if config.chunk_long_sequences else base_max_length,
                "n_epochs": base_n_epochs,
                "batch_size": base_batch_size,
                "chunk_context": 16,
                "predict_batch_size": 256 if config.float_16_predict else 48,
                "mixed_precision": False,
                "float_16_predict": False,
                "lr": base_learning_rate,
            }
        elif config.optimize_for.lower() == "accuracy_fp16":
            overrides = {
                "max_length": base_max_length,
                "n_epochs": base_n_epochs,
                "batch_size": 12,
                "chunk_context": None,
                "predict_batch_size": 40,
                "mixed_precision": True,
                "float_16_predict": True,
                "lr": base_learning_rate * 6,
            }

        elif config.optimize_for.lower() == "predict_speed_fp16":
            overrides = {
                "max_length": 128 if config.chunk_long_sequences else base_max_length,
                "n_epochs": base_n_epochs,
                "batch_size": 64,
                "chunk_context": 16,
                "predict_batch_size": 256,
                "mixed_precision": True,
                "float_16_predict": True,
                "lr": base_learning_rate * 32,
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
        **BERT_BASE_PARAMS,
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
        **BERT_LARGE_PARAMS,
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
        **BERT_LARGE_PARAMS,
        "base_model_path": os.path.join("bert", "bert_wwm_large_cased-v2.jl"),
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
        **BERT_BASE_PARAMS,
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
    is_roberta = True
    settings = {
        **BERT_BASE_PARAMS,
        "epsilon": 1e-8,
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
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]

    @classmethod
    def get_encoder(cls, config=None, **kwargs):
        roberta_filename = config.base_model_path.rpartition("/")[-1]
        missing_mask_token = roberta_filename in (
            "roberta-model-sm.jl",
            "roberta-model-lg.jl",
        )
        if config is not None and not missing_mask_token:
            return cls.encoder(**kwargs)
        else:
            return RoBERTaEncoder(**kwargs)


class FusedRoBERTa(RoBERTa):
    featurizer = fused_featurizer(bert_featurizer)
    settings = dict(RoBERTa.settings)
    settings.update(
        {"max_length": 2048, "num_fusion_shards": 4, "chunk_long_sequences": False}
    )


class DocRep(_BaseBert):
    encoder = RoBERTaEncoderV2
    featurizer = bert_featurizer
    get_context_fn = get_context_doc_rep
    is_roberta = True
    settings = dict(RoBERTa.settings)
    settings.update(
        {
            "lr": 1e-4,
            "context_injection": True,
            "reading_order_removed": True,
            "context_channels": 192,
            "crf_sequence_labeling": False,
            "context_dim": 4,
            "default_context": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "use_auxiliary_info": True,
            "low_memory_mode": True,
            "base_model_path": os.path.join("bert", "doc_rep_v1.jl"),
        }
    )
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "doc_rep_v1.jl"
            ),
            "url": urljoin(BERT_BASE_URL, "doc_rep_v1.jl"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "dict.txt"),
            "url": urljoin(ROBERTA_BASE_URL, "dict.txt"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]

    @classmethod
    def get_encoder(cls, config=None, **kwargs):
        return cls.encoder(**kwargs)


class FusedDocRep(DocRep):
    featurizer = fused_featurizer(bert_featurizer)
    settings = dict(DocRep.settings)
    settings.update(
        {"max_length": 2048, "num_fusion_shards": 4, "chunk_long_sequences": False}
    )


class RoBERTaLarge(RoBERTa):
    encoder = RoBERTaEncoderV2
    is_roberta = True
    featurizer = bert_featurizer
    settings = {
        **BERT_LARGE_PARAMS,
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
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]


ZuckerBERT = RoBERTa


class DistilBERT(_BaseBert):
    encoder = DistilBERTEncoder
    featurizer = bert_featurizer
    settings = {
        **DISTIL_BERT_PARAMS,
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
    is_roberta = True

    settings = {
        **DISTIL_BERT_PARAMS,
        "base_model_path": os.path.join("bert", "distilroberta.jl"),
    }
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "distilroberta.jl"
            ),
            "url": urljoin(BERT_BASE_URL, "distilroberta.jl"),
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
        },
    ]


class LayoutLM(_BaseBert):
    encoder = LayoutLMEncoder
    featurizer = layoutlm_featurizer
    get_context_fn = get_context_layoutlm
    settings = {
        **BERT_BASE_PARAMS,
        "epsilon": 1e-8,
        "lr": 1e-4,
        "context_injection": True,
        "crf_sequence_labeling": False,
        "context_dim": 4,
        "default_context": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "use_auxiliary_info": True,
        "low_memory_mode": True,
        "base_model_path": os.path.join("bert", "layoutlm-base-uncased.jl"),
        "include_bos_eos": False,
    }
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "layoutlm-base-uncased.jl"
            ),
            "url": urljoin(LAYOUTLM_BASE_URL, "layoutlm-base-uncased.jl"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "layoutlm_vocab.txt"
            ),
            "url": urljoin(LAYOUTLM_BASE_URL, "layoutlm_vocab.txt"),
        },
    ]


class XDocBase(_BaseBert):
    featurizer = xdoc_featurizer
    encoder = RoBERTaEncoderXDoc
    get_context_fn = get_context_layoutlm
    is_roberta = False
    settings = {
        **BERT_BASE_PARAMS,
        "epsilon": 1e-8,
        "lr": 1e-4,
        "context_injection": True,
        "crf_sequence_labeling": False,
        "context_dim": 4,
        "default_context": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "use_auxiliary_info": True,
        "low_memory_mode": True,
        "base_model_path": os.path.join("bert", "XDoc.jl"),
        "act_fn": "hf_gelu",
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "XDoc.jl"),
            "url": urljoin(ROBERTA_BASE_URL, "XDoc.jl"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "dict.txt"),
            "url": urljoin(ROBERTA_BASE_URL, "dict.txt"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]

    @classmethod
    def get_encoder(cls, config=None, **kwargs):
        return cls.encoder(**kwargs)


class TableRoBERTa(_BaseBert):
    encoder = RoBERTaEncoderV2
    is_roberta = True
    featurizer = table_roberta_featurizer_twinbert
    settings = {
        **BERT_BASE_PARAMS,
        # Just incase all cells fall into the same buckets this -8 allows us to pack the batches much tighter once we add EOS and BOS
        "max_length": 2048 - 8,
        "table_batching": True,
        "chunk_tables": True,
        "batch_size": 2,  # When table batching is true this becomes a nominal batch size. Very long docs have batch size = 1
        "class_weights": None,
        "n_epochs": 24,
        "epsilon": 1e-8,
        "lr": 1e-4,
        "context_injection": True,
        "crf_sequence_labeling": True,
        "context_dim": 4,
        "default_context": {
            "end_col": -1,
            "start_col": -1,
            "end_row": -1,
            "start_row": -1,
        },
        "use_auxiliary_info": True,
        "low_memory_mode": True,
        "base_model_path": os.path.join("bert", "roberta-table-twinbert-v1.jl"),
        "bert_use_pooler": False,
        "chunk_long_sequences": False,
        "include_bos_eos": False,
        "permit_uninitialized": r"mixing_fn_|pos_|kernel|bias",  # TODO: this can be refined.
    }
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta-table-twinbert-v1.jl"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta-table-twinbert-v1.jl"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "dict.txt"),
            "url": urljoin(ROBERTA_BASE_URL, "dict.txt"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]

import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.bert.encoder import (
    BERTEncoder,
    BERTEncoderMultuilingal,
    BERTEncoderLarge,
    DistilBERTEncoder
)

from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder
from finetune.base_models.bert.featurizer import bert_featurizer
from finetune.util.download import BERT_BASE_URL, GPT2_BASE_URL, ROBERTA_BASE_URL, FINETUNE_BASE_FOLDER


class BERTModelCased(SourceModel):
    is_bidirectional = True
    encoder = BERTEncoder
    featurizer = bert_featurizer
    settings = {
        "n_embed": 768,
        "n_epochs": 8,
        "n_heads": 12,
        "n_layer": 12,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_intermediate_size": 3072,
        "base_model_path": os.path.join("bert", "bert_small_cased.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_small_cased.jl", "vocab.txt"]
    ]


class BERTModelLargeCased(SourceModel):
    is_bidirectional = True
    encoder = BERTEncoderLarge
    featurizer = bert_featurizer
    settings = {
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
        "base_model_path": os.path.join("bert", "bert_large_cased.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_large_cased.jl", "vocab_large.txt"]
    ]


class BERTModelLargeWWMCased(SourceModel):
    is_bidirectional = True
    encoder = BERTEncoderLarge
    featurizer = bert_featurizer
    settings = {
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
        "base_model_path": os.path.join("bert", "bert_wwm_large_cased.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_wwm_large_cased.jl", "vocab_large.txt"]
    ]


class BERTModelMultilingualCased(SourceModel):
    is_bidirectional = True
    encoder = BERTEncoderMultuilingal
    featurizer = bert_featurizer
    settings = {
        "n_embed": 768,
        "n_epochs": 8,
        "n_heads": 12,
        "n_layer": 12,
        "act_fn": "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        "bert_intermediate_size": 3072,
        "base_model_path": os.path.join("bert", "bert_small_multi_cased.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", filename),
            "url": urljoin(BERT_BASE_URL, filename),
        }
        for filename in ["bert_small_multi_cased.jl", "vocab_multi.txt"]
    ]


class RoBERTa(SourceModel):
    is_bidirectional = True
    encoder = RoBERTaEncoder
    featurizer = bert_featurizer
    settings = {
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
        "max_length": 512,
        "base_model_path": os.path.join("bert", "roberta-model-sm.jl"),
    }
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta-model-sm.jl"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta-model-sm.jl"),
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

class RoBERTaLarge(SourceModel):
    is_bidirectional = True
    encoder = RoBERTaEncoder
    featurizer = bert_featurizer
    settings = {
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
        "max_length": 512,
        "base_model_path": os.path.join("bert", "roberta-model-lg.jl"),
    }
    required_files = [
        {
            "file": os.path.join(
                FINETUNE_BASE_FOLDER, "model", "bert", "roberta-model-lg.jl"
            ),
            "url": urljoin(ROBERTA_BASE_URL, "roberta-model-lg.jl"),
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

   
ZuckerBERT = RoBERTa


class DistilBERT(SourceModel):
    is_bidirectional = True
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


class DistilRoBERTa(SourceModel):
    is_bidirectional = True
    encoder = RoBERTaEncoder
    featurizer = bert_featurizer
    settings = {
        "max_length": 512,
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

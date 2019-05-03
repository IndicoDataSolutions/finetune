import os

from finetune.base_models import SourceModel
from finetune.base_models.bert.encoder import BERTEncoder, BERTEncoderMultuilingal, BERTEncoderLarge
from finetune.base_models.bert.featurizer import bert_featurizer


class BERTModelCased(SourceModel):
    is_bidirectional = True
    encoder = BERTEncoder
    featurizer = bert_featurizer
    settings = {
        'n_embed': 768,
        'n_epochs': 8,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        'bert_intermediate_size': 3072,
        "base_model_path": os.path.join("bert", "bert_small_cased.jl"),
    }


class BERTModelLargeCased(SourceModel):
    is_bidirectional = True
    encoder = BERTEncoderLarge
    featurizer = bert_featurizer
    settings = {
        'n_embed': 1024,
        'n_epochs': 8,
        'n_heads': 16,
        'n_layer': 24,
        'num_layers_trained': 24,
        'act_fn': "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        'bert_intermediate_size': 4096,
        "base_model_path": os.path.join("bert", "bert_large_cased.jl"),
    }


class BERTModelMultilingualCased(SourceModel):
    is_bidirectional = True
    encoder = BERTEncoderMultuilingal
    featurizer = bert_featurizer
    settings = {
        'n_embed': 768,
        'n_epochs': 8,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
        "lr_warmup": 0.1,
        "lr": 1e-5,
        "l2_reg": 0.01,
        'bert_intermediate_size': 3072,
        "base_model_path": os.path.join("bert", "bert_small_multi_cased.jl"),
    }

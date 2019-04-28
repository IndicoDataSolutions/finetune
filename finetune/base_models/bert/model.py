import os

from finetune.base_models import SourceModel
from finetune.base_models.bert.encoder import BERTEncoder, BERTEncoderMultuilingal, BERTEncoderLarge
from finetune.base_models.bert.featurizer import bert_featurizer


class BERTModelCased(SourceModel):
    encoder = BERTEncoder
    featurizer = bert_featurizer
    settings = {
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
        "lr": 5e-6,
        'bert_intermediate_size': 3072,
        "base_model_path": os.path.join("bert", "bert_small_cased.jl")
    }


class BERTModelLargeCased(SourceModel):
    encoder = BERTEncoderLarge
    featurizer = bert_featurizer
    settings = {
        'n_embed': 1024,
        'n_heads': 16,
        'n_layer': 24,
        'act_fn': "gelu",
        "lr": 5e-6,
        'bert_intermediate_size': 4096,
        "base_model_path": os.path.join("bert", "bert_large_cased.jl")
    }


class BERTModelMultilingualCased(SourceModel):
    encoder = BERTEncoderMultuilingal
    featurizer = bert_featurizer
    settings = {
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
        "lr": 5e-6,
        'bert_intermediate_size': 3072,
        "base_model_path": os.path.join("bert", "bert_small_multi_cased.jl")
    }

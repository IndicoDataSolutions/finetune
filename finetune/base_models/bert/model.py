import os

from finetune.base_models import SourceModel
from finetune.base_models.bert.encoder import BERTEncoder, BERTEncoderMultuilingal
from finetune.base_models.bert.featurizer import bert_featurizer


class BERTModelCased(SourceModel):
    encoder = BERTEncoder
    featurizer = bert_featurizer
    settings = {
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
        'bert_intermediate_size': 3072,
        "base_model_path": os.path.join("bert", "bert_small_cased.jl")
    }

class BERTModelMultilingualCased(SourceModel):
    encoder = BERTEncoderMultuilingal
    featurizer = bert_featurizer
    settings = {
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
        'bert_intermediate_size': 3072,
        "base_model_path": os.path.join("bert", "bert_small_multi_cased.jl")
    }
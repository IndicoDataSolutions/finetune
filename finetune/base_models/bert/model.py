import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.bert.encoder import BERTEncoder, BERTEncoderMultuilingal, BERTEncoderLarge
from finetune.base_models.bert.featurizer import bert_featurizer
from finetune.util.download import BERT_BASE_URL, FINETUNE_BASE_FOLDER


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
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'bert', filename),
            'url': urljoin(BERT_BASE_URL, filename)
        } for filename in [
            "bert_small_cased.jl", "vocab.txt"
        ]
    ]


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
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'bert', filename),
            'url': urljoin(BERT_BASE_URL, filename)
        } for filename in [
            "bert_large_cased.jl", "vocab_large.txt"
        ]
    ]


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
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'bert', filename),
            'url': urljoin(BERT_BASE_URL, filename)
        } for filename in [
            "bert_small_multi_cased.jl", "vocab_multi.txt"
        ]
    ]

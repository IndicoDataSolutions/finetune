import os

from finetune.base_models import SourceModel
from finetune.base_models.gpc.encoder2 import GPCEncoder
from finetune.base_models.gpc.featurizer import featurizer


class GPCModel(SourceModel):
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        'n_embed': 512,
        "base_model_path": "fresh_start.jl",
        'n_layer': 4,
        'num_layers_trained':4
    }


class GPCModelFP16(SourceModel):
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        'n_embed': 512,
        "base_model_path": "fresh_start.jl",
        "optimizer": "Adafactor",
        "use_fp16": True,
        "scale_loss": True,
        'n_layer': 4,
        'num_layers_trained':4,
        "stochastic_tokens":True
    }

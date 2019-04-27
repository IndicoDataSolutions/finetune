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
        'n_layer': 12,
        'num_layers_trained':12,
        'n_epochs': 8,
        'prefit_init': True,
        'batch_size': 8,
        'feat_mode': 'max_state',
        'l2_reg': 0.0,
        'lr': 0.001,
        'lr_warmup': 0.0
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
        'n_layer': 12,
        'num_layers_trained':12,
        "stochastic_tokens":True
    }

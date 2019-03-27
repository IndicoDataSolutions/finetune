import os

from finetune.base_models import SourceModel
from finetune.base_models.gpc.encoder import GPTEncoder
from finetune.base_models.gpc.featurizer import featurizer


class GPCModel(SourceModel):
    encoder = GPTEncoder
    featurizer = featurizer
    settings = {
        'n_embed': 256,
        "base_model_path": os.path.join("gpt", "model-lg.jl"),
        'n_layer': 12,
    }


class GPCModelFP16(SourceModel):
    encoder = GPTEncoder
    featurizer = featurizer
    settings = {
        'n_embed': 256,
        "base_model_path": os.path.join("gpt", "model-lg.jl"),
        "optimizer": "Adafactor",
        "use_fp16": True,
        "scale_loss": True,
        'n_layer': 12,
    }

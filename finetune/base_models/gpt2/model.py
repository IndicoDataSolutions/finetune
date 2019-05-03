import os

from finetune.base_models import SourceModel
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.gpt2.featurizer import gpt2_featurizer


class GPT2Model(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = gpt2_featurizer
    settings = {
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'l2_reg': 0.001,
        'act_fn': "gelu",
        'interpolate_pos_embed': False,
        'base_model_path':  os.path.join("gpt2", "model-sm.jl"),
    }

import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt.featurizer import gpt_featurizer
from finetune.util.download import GPT_BASE_URL, FINETUNE_BASE_FOLDER


class GPTModel(SourceModel):
    is_bidirectional = False
    encoder = GPTEncoder
    featurizer = gpt_featurizer
    settings = {
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
        "base_model_path": os.path.join("gpt", "model-lg.jl"),
    }
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'gpt', filename),
            'url': urljoin(GPT_BASE_URL, filename)
        }
        for filename in ['encoder.json', 'vocab.bpe', 'model-lg.jl']
    ]


class GPTModelSmall(GPTModel):
    is_bidirectional = False
    settings = {
        'n_embed': 512,
        'n_heads': 8,
        'n_layer': 6,
        'act_fn': "gelu",
        'num_layers_trained': 6,
        'base_model_path': os.path.join("gpt", "model-sm.jl"),
    }
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'gpt', filename),
            'url': urljoin(GPT_BASE_URL, filename)
        }
        for filename in ['encoder.json', 'vocab.bpe', 'model-sm.jl']
    ]


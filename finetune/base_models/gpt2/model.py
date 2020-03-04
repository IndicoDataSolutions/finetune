import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.gpt2.featurizer import gpt2_featurizer
from finetune.util.download import GPT2_BASE_URL, FINETUNE_BASE_FOLDER


class GPT2Model(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = gpt2_featurizer
    settings = {
        'max_length': 1024,
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'l2_reg': 0.001,
        'act_fn': "gelu",
        'base_model_path':  os.path.join("gpt2", "model-sm.jl")
    }
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'gpt2', filename),
            'url': urljoin(GPT2_BASE_URL, filename)
        }
        for filename in ['encoder.json', 'vocab.bpe', 'model-sm.jl']
    ]


class GPT2Model345(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = gpt2_featurizer
    settings = {
        'max_length': 1024,
        'n_embed': 1024,
        'n_heads': 16,
        'n_layer': 24,
        'num_layers_trained': 24,
        'l2_reg': 0.001,
        'act_fn': "gelu",
        'base_model_path':  os.path.join("gpt2", "model-med.jl")
    }
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'gpt2', filename),
            'url': urljoin(GPT2_BASE_URL, filename)
        }
        for filename in ['encoder.json', 'vocab.bpe', 'model-med.jl']
    ]


class GPT2Model762(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = gpt2_featurizer
    settings = {
        'max_length': 1024,
        'n_embed': 1280,
        'n_heads': 20,
        'n_layer': 36,
        'num_layers_trained': 36,
        'l2_reg': 0.001,
        'act_fn': "gelu",
        'base_model_path':  os.path.join("gpt2", "model-lg.jl")
    }
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'gpt2', filename),
            'url': urljoin(GPT2_BASE_URL, filename)
        }
        for filename in ['encoder.json', 'vocab.bpe', 'model-lg.jl']
    ]



class GPT2Model1558(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = gpt2_featurizer

    settings = {
        'max_length': 1024,
        'n_embed': 1600,
        'n_heads': 25,
        'n_layer': 48,
        'num_layers_trained': 48,
        'l2_reg': 0.001,
        'act_fn': "gelu",
        'base_model_path':  os.path.join("gpt2", "model-xl.jl")
    }
    required_files = [
        {
            'file': os.path.join(FINETUNE_BASE_FOLDER, 'model', 'gpt2', filename),
            'url': urljoin(GPT2_BASE_URL, filename)
        }
        for filename in ['encoder.json', 'vocab.bpe', 'model-xl.jl']
    ]

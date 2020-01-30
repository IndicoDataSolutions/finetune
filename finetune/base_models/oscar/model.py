import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.oscar.encoder import GPCEncoder
from finetune.base_models.oscar.featurizer import featurizer
from finetune.util.download import FINETUNE_BASE_FOLDER, OSCAR_BASE_URL

REQUIRED_FILES = [
    {
        "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "oscar", filename),
        "url": urljoin(OSCAR_BASE_URL, filename),
    }
    for filename in ["encoder.model", "encoder.vocab", "fresh_start.jl", "oscar23.jl"]
]

BASE_OSCAR_SETTINGS = {
    'n_embed': 768,
    "base_model_path": os.path.join("oscar", "oscar23.jl"),
    'n_layer': 12,
    'num_layers_trained': 12,
    'lr_warmup': 0.1,
    "distribution_strategy": "mirrored",
    'oscar_feat_mode': 'clf_tok',
    "l2_reg": 0.05,
    "batch_size": 8,
    "lr": 0.0001,
    "n_epochs": 2,
    "resid_p_drop": 0.01,
    "max_grad_norm": 1.0,
    "embed_p_drop": 0.01,
    "lr_schedule": "warmup_linear",
    "lm_loss_coef": 0.5,
}


class GPCModel(SourceModel):
    is_bidirectional = False
    encoder = GPCEncoder
    featurizer = featurizer
    settings = BASE_OSCAR_SETTINGS
    required_files = REQUIRED_FILES


class GPCModelFP16(SourceModel):
    is_bidirectional = False
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        **GPCModel.settings,
        "use_fp16": True,
        "low_memory_mode": True,
        "scale_loss": True,
    }
    required_files = REQUIRED_FILES


class GPCModelFP16Pretrain(SourceModel):
    is_bidirectional = False
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        **GPCModelFP16.settings,
        "base_model_path": os.path.join("oscar", "fresh_start.jl"),
        "optimizer": "Adafactor",
        "cache_weights_to_file": True,
        "lr": 1e-4,
        "batch_size": 60,
        "keep_best_model": True,
        "val_interval": 1000,
        "val_size": 2000,
        "lr_schedule": "exp_decay_oscar",
        "n_epochs": 4,
    }
    required_files = REQUIRED_FILES

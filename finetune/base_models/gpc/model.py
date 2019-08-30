import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.gpc.encoder import GPCEncoder
from finetune.base_models.gpc.featurizer import featurizer
from finetune.util.download import FINETUNE_BASE_FOLDER, OSCAR_BASE_URL

REQUIRED_FILES = [
    {
        "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "oscar", filename),
        "url": urljoin(OSCAR_BASE_URL, filename),
    }
    for filename in ["encoder.model", "encoder.vocab", "fresh_start.jl"]
]

BASE_OSCAR_SETTINGS = {
    'n_embed': 768,
    "base_model_path": os.path.join("oscar", "fresh_start.jl"),
    'n_layer': 12,
    'num_layers_trained': 12,
    'lr_warmup': 0.1,
    "xla": True,
    "use_mirrored_distribution": True,
    'feat_mode': 'clf_tok',
}


class GPCModel(SourceModel):
    is_bidirectional = False
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        **BASE_OSCAR_SETTINGS,
        'n_epochs': 3,
        'batch_size': 3,
        'l2_reg': 0.1,
        'lr': 0.0001,
    }
    required_files = REQUIRED_FILES


class GPCModelFP16(SourceModel):
    is_bidirectional = False
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        **BASE_OSCAR_SETTINGS,
        "optimizer": "Adafactor",
        "use_fp16": True,
        "scale_loss": True,
        "low_memory_mode": True,
        "cache_weights_to_file": True,
        "lr": 0.01
    }
    required_files = REQUIRED_FILES

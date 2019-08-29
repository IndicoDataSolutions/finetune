from finetune.base_models import SourceModel
from finetune.base_models.gpc.encoder import GPCEncoder
from finetune.base_models.gpc.featurizer import featurizer


class GPCModel(SourceModel):
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        'n_embed': 768,
        "base_model_path": "conv_base_30jun.jl",
        'n_layer': 12,
        'num_layers_trained':12,
        'n_epochs': 3,
        'prefit_init': True,
        'batch_size': 3,
        'feat_mode': 'clf_tok',
        'l2_reg': 0.1,
        'lr': 0.0001,
        'lr_warmup': 0.1,
        "xla": True,
        "use_mirrored_distribution": True,
    }


class GPCModelFP16(SourceModel):
    encoder = GPCEncoder
    featurizer = featurizer
    settings = {
        "feat_mode": "clf_tok",
        'n_embed': 768,
        "base_model_path": "fresh_start.jl",
        "optimizer": "Adafactor",
        "use_fp16": True,
        "scale_loss": True,
        'n_layer': 12,
        'num_layers_trained': 12,
        "stochastic_tokens": False,
        "low_memory_mode": True,
        "cache_weights_to_file": True,
        "use_mirrored_distribution": True,
        "xla": True,
        "lr": 0.01
    }

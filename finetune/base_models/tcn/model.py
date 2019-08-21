import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.tcn.featurizer import tcn_featurizer
from finetune.util.download import GPT2_BASE_URL, FINETUNE_BASE_FOLDER


class TCNModel(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = tcn_featurizer
    settings = {
        "n_epochs": 5,
        "n_embed_featurizer": 768,
        "n_layer": 3,
        "num_layers_trained": 3,
        "n_filter": 10,
        "n_embed": 10,
        "seq_num_heads": 10,
        "kernel_size": 5,
        "act_fn": "gelu",
        "train_embeddings": False,
        "lr": 0.001,
        "base_model_path": os.path.join("gpt2", "model-sm.jl"),
        "n_context_embed": 10
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "gpt2", filename),
            "url": urljoin(GPT2_BASE_URL, filename),
        }
        for filename in ["encoder.json", "vocab.bpe", "model-sm.jl"]
    ]

import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.textcnn.featurizer import textcnn_featurizer
from finetune.util.download import GPT2_BASE_URL, FINETUNE_BASE_FOLDER


class TextCNNModel(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = textcnn_featurizer
    kernel_sizes = [2, 3, 4]
    settings = {
        "n_embed_featurizer": 768,
        "n_layer": 1,
        "n_epochs": 100,
        "keep_best_model": True,
        "early_stopping_steps": 100,
        "val_size": "auto",
        "num_layers_trained": 1,
        "kernel_sizes": kernel_sizes,
        "num_filters_per_size": 2,
        "n_embed": len(kernel_sizes) * 2,
        "act_fn": "gelu",
        "train_embeddings": True,
        "lr": 0.01,
        "seq_num_heads": len(kernel_sizes) * 2,
        "base_model_path": os.path.join("gpt2", "model-sm.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "gpt2", filename),
            "url": urljoin(GPT2_BASE_URL, filename),
        }
        for filename in ["encoder.json", "vocab.bpe", "model-sm.jl"]
    ]

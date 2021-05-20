import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.textcnn.featurizer import textcnn_featurizer
from finetune.util.download import GPT2_BASE_URL, FINETUNE_BASE_FOLDER


class TextCNNModel(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = textcnn_featurizer
    kernel_sizes = [2, 4, 8]
    settings = {
        "n_embed_featurizer": 768,
        "n_layer": 1,
        "n_epochs": 8,
        "keep_best_model": False,
        "early_stopping_steps": None,
        "val_size": 0,
        "chunk_long_sequences": False,
        "num_layers_trained": 1,
        "kernel_sizes": kernel_sizes,
        "num_filters_per_size": 256,
        "n_embed": len(kernel_sizes) * 256,
        "act_fn": "gelu",
        "train_embeddings": True,
        "lr": 2e-3,
        "seq_num_heads": len(kernel_sizes) * 2,
        "base_model_path": os.path.join("gpt2", "model-sm.jl"),
        "permit_uninitialized": r"conv[0-9]+",
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "gpt2", filename),
            "url": urljoin(GPT2_BASE_URL, filename),
        }
        for filename in ["encoder.json", "vocab.bpe", "model-sm.jl"]
    ]
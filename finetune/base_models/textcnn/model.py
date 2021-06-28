import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoderV2
from finetune.base_models.textcnn.featurizer import textcnn_featurizer
from finetune.util.download import (
    GPT2_BASE_URL,
    ROBERTA_BASE_URL,
    FINETUNE_BASE_FOLDER,
)

KERNEL_SIZES = [2, 4, 8]
TEXTCNN_BASE_PARAMS = {
    "n_embed_featurizer": 768,
    "n_layer": 1,
    "n_epochs": 8,
    "keep_best_model": False,
    "early_stopping_steps": None,
    "val_size": 0,
    "chunk_long_sequences": False,
    "num_layers_trained": 1,
    "kernel_sizes": KERNEL_SIZES,
    "num_filters_per_size": 256,
    "n_embed": len(KERNEL_SIZES) * 256,
    "act_fn": "gelu",
    "train_embeddings": True,
    "lr": 2e-3,
    "seq_num_heads": len(KERNEL_SIZES) * 2,
    "permit_uninitialized": r"conv[0-9]+",
    # For reasonable memory consumption
    "max_length": 25000,
    "predict_batch_size": 4,
}


class TextCNNModel(SourceModel):
    is_bidirectional = False
    encoder = GPT2Encoder
    featurizer = textcnn_featurizer
    settings = {
        **TEXTCNN_BASE_PARAMS,
        "base_model_path": os.path.join("gpt2", "model-sm.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "gpt2", filename),
            "url": urljoin(GPT2_BASE_URL, filename),
        }
        for filename in ["encoder.json", "vocab.bpe", "model-sm.jl"]
    ]


class FastTextCNNModel(SourceModel):
    """Uses RobertaEncoderV2 encoder for fast tokenization"""
    is_bidirectional = False
    encoder = RoBERTaEncoderV2
    featurizer = textcnn_featurizer
    settings = {
        **TEXTCNN_BASE_PARAMS,
        "base_model_path": os.path.join("bert", "roberta-model-sm-v2.jl"),
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "roberta-model-sm-v2.jl"),
            "url": urljoin(ROBERTA_BASE_URL, "roberta-model-sm-v2.jl"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "dict.txt"),
            "url": urljoin(ROBERTA_BASE_URL, "dict.txt"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "roberta_vocab.bpe"),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_vocab.bpe"),
        },
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "bert", "roberta_encoder.json"),
            "url": urljoin(ROBERTA_BASE_URL, "roberta_encoder.json"),
        },
    ]

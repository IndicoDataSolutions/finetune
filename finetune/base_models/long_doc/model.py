import os
from urllib.parse import urljoin

import tensorflow as tf

from finetune.base_models import SourceModel
from finetune.base_models.long_doc.featurizer import long_doc_featurizer

from finetune.base_models.long_doc.encoder import LongDocEncoder
from finetune.util.download import GPT2_BASE_URL, FINETUNE_BASE_FOLDER


class LongDocModel(SourceModel):

    @classmethod
    def get_token_type_shape(cls):
        return tf.float32, [None, cls.settings["n_embed_featurizer"]]
        
    is_bidirectional = True
    requires_pre_tokenized = True
    encoder = LongDocEncoder
    featurizer = long_doc_featurizer
    settings = {
        "batch_size": 32,
        "n_epochs": 200,
        "n_embed_featurizer": 300,
        "n_layer": 3,
        "num_layers_trained": 3,
        "n_filter": 10,
        "n_embed": 10,
        "kernel_size": 3,
        "val_size": "auto",
        "keep_best_model": True,
        "train_embeddings": False,
        "lr": 0.3,
        "base_model_path": None,
        "reweighting": "sqrt"
    }
    required_files = []

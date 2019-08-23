import os
from urllib.parse import urljoin

from finetune.util.download import FINETUNE_BASE_FOLDER, GLOVE_BASE_URL
from finetune.base_models.glove.encoder import GLOVEEncoder
from finetune.base_models.glove.featurizer import glove_featurizer
from finetune.base_models import SourceModel

class GloveModel(SourceModel):
    is_bidirectional = False
    encoder = GLOVEEncoder
    featurizer = glove_featurizer
    settings = {
        "base_model_path": os.path.join("glove", "glove_embeddings.jl"),
        'n_layer': 1,
        'num_layers_trained': 1,
        'train_embeddings': False,
        'n_embed': 300,
        'seq_num_heads': 15,
        'low_memory_mode': True
    }
    required_files = [
        {
            "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "glove", filename),
            "url": urljoin(GLOVE_BASE_URL, filename),
        }
        for filename in ["glove_embeddings.jl", "encoder.json"]
    ]


    
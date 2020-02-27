import os
from urllib.parse import urljoin

from finetune.base_models import SourceModel
from finetune.base_models.glove.encoder import GloveEncoder
from finetune.base_models.glove.featurizer import glove_featurizer
from finetune.util.download import GLOVE_BASE_URL, FINETUNE_BASE_FOLDER

class GloveModel(SourceModel):
    is_bidirectional = False
    encoder = GloveEncoder
    featurizer = glove_featurizer
    settings = {
        'batch_size': 128,
        'n_embed_featurizer': 300,
        'n_epochs': 100,
        'keep_best_model': True,
        'early_stopping_steps': 100,
        'val_size': "auto",
        'n_layer': 1,
        'num_layers_trained': 1,
        'n_embed': 300,
        'train_embeddings': False,
        'lr': 0.01,
        'seq_num_heads': 5,
        'base_model_path': os.path.join('glove', 'model.jl')
    }
    required_files = (
        [
            {
                "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "glove", "vocab.spacy", filename),
                "url": urljoin(GLOVE_BASE_URL, filename)
            } for filename in ["key2row", "lexemes.bin", "strings.json"]
        ] + 
        [
            {
                "file": os.path.join(FINETUNE_BASE_FOLDER, "model", "glove", filename),
                "url": urljoin(GLOVE_BASE_URL, filename)
            } for filename in ["tokenizer.spacy", 'model.jl']
        ]
    )

      

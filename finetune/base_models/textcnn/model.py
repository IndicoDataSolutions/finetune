import os
from finetune.base_models import SourceModel
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.textcnn.featurizer import textcnn_featurizer


class TextCNNModel(SourceModel):
    is_bidirectional = False
    encoder = GPTEncoder
    featurizer = textcnn_featurizer
    kernel_sizes = [2, 3, 4]
    settings = {
        'n_embed_featurizer': 768,
        'n_layer': 1,
        'num_layers_trained': 1,
        'kernel_sizes': kernel_sizes,
        'num_filters_per_size': 2,
        'n_embed': len(kernel_sizes) * 2,
        'act_fn': "gelu",
        'train_embeddings': False,
        'lr': .01,
        'seq_num_heads': len(kernel_sizes) * 2,
        'base_model_path': os.path.join("gpt", "model-lg.jl"),
    }

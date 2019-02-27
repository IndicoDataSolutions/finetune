from finetune.base_models import SourceModel
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt.featurizer import gpt_featurizer

class GPTModel(SourceModel):

    encoder = GPTEncoder
    featurizer = gpt_featurizer
    settings = {
        'n_embed': 768,
        'n_heads': 12,
        'n_layer': 12,
        'act_fn': "gelu",
    }

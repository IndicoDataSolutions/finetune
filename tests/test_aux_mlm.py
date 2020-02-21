import unittest
import json
import joblib
import tensorflow as tf
import numpy as np

from finetune.base_models import RoBERTa
from finetune import MaskedLanguageModel

from copy import deepcopy

class TestAuxMLM(unittest.TestCase):
    def setUp(self):
        aux_config = {
            'use_auxiliary_info': True,
            'context_dim': 4,
            'n_context_embed_per_channel': 48,
            'context_in_base_model': True,
            'n_layers_with_aux': -1,
            'default_context': {'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
        }

        no_aux_config = deepcopy(aux_config)
        no_aux_config['use_auxiliary_info'] = False
        no_aux_config['context_dim'] = 0
        no_aux_config['context_in_base_model'] = False
        no_aux_config['n_layers_with_aux'] = 0

        self.model = MaskedLanguageModel(base_model=RoBERTa, n_epochs=1, lr=0.0, **no_aux_config)
        self.model_with_pos = MaskedLanguageModel(base_model=RoBERTa, n_epochs=1, lr=0.0, **aux_config)
        self.text = []
        self.context = []
        with open('tests/sample_text.ndjson') as f:
            for line in f:
                text, context = json.loads(line)
                self.text.append(text)
                self.context.append(context)

    def test_compare_activations(self):
        seq_features_pos = self.model_with_pos.featurize_sequence(self.text[:5], context=self.context[:5])
        seq_features = self.model.featurize_sequence(self.text[:5])

        num_pos_feats = self.model_with_pos.config['n_context_embed_per_channel']*self.model_with_pos.config['context_dim']
        print(seq_features.shape)
        print(seq_features_pos.shape)
        np.save('seq_features.npy', seq_features)
        np.save('seq_features_pos.npy', seq_features_pos)
        for act1, act2 in zip(seq_features, seq_features_pos):
            np.testing.assert_array_almost_equal(
                act1,
                act2[:,:-num_pos_feats],
                decimal=3)

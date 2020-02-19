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
        # no_aux_config = {
        #     'use_auxiliary_info': False,
        #     'context_dim': 0,
        #     'n_context_embed_per_channel': 48,
        #     'context_in_base_model': False,
        #     'n_layers_with_aux': 0,
        #     'default_context': {'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
        # }
        self.model = MaskedLanguageModel(base_model=RoBERTa, n_epochs=1, lr=0.0, **no_aux_config)
        self.model_with_pos = MaskedLanguageModel(base_model=RoBERTa, n_epochs=1, lr=0.0, **aux_config)
        self.text = []
        self.context = []
        with open('tests/sample_text.ndjson') as f:
            for line in f:
                text, context = json.loads(line)
                self.text.append(text)
                self.context.append(context)
        print(type(self.text))
        print(type(self.context))
        print(len(self.text))
        print(len(self.context))

        # self.text = [
        #     'apple',
        #     # 'banana',
        #     'orange',
        #     'orange'
        # ]
        # self.context = [
        #     [{'left': 0, 'right': 0, 'top': 0, 'bottom': 3, 'start': 0, 'end': 5}],
        #     [{'left': 0, 'right': 10, 'top': 0, 'bottom': 0, 'start': 0, 'end': 6}],
        #     [{'left': 0, 'right': 0, 'top': 30, 'bottom': 100, 'start': 0, 'end': 6}]
        # ]

        # self.text = [
        #     # ' '.join(['apple'] * 256),
        #     # 'banana',
        #     ' '.join(['orange'] * 256),
        #     ' '.join(['orange'] * 256),
        #     ' '.join(['orange'] * 256)
        # ]
        # self.context = [
        #     [{'left': 0, 'right': 0, 'top': 0, 'bottom': 3, 'start': 0, 'end': 6}] * 256,
        #     [{'left': 0, 'right': 10, 'top': 0, 'bottom': 0, 'start': 0, 'end': 6}] * 256,
        #     [{'left': 0, 'right': 0, 'top': 30, 'bottom': 100, 'start': 0, 'end': 6}] * 256
        # ]
        # for context in self.context:
        #     prev_start = 0
        #     prev_end = 6
        #     for c in context:
        #         c['start'] = prev_start + 7
        #         c['end'] = prev_end + 7
        #         prev_start = c['start']
        #         prev_end = c['end']


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
    
    def test_variables_used(self):
        self.model.fit(self.text)
    #     base_model_file = self.model.config.base_model_path
    #     base_model_vars = joblib.load(base_model_file)
    #     # model_vars = [w.name for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
    #     # print(set(base_model_vars) - set(model_vars))
    #     # print(set(model_vars) - set(base_model_vars))
    #     estimator, _ = self.model.get_estimator()
    #     model_vars = estimator.get_variable_names()
    #     print(model_vars)
    #     self.assertFalse(set(base_model_vars) - set(model_vars))
    #     self.assertFalse(set(model_vars) - set(base_model_vars))

    #     print('******************')
    #     self.model_with_pos.fit(self.text)
    #     model_vars = [w.name for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
    #     print(set(base_model_vars) - set(model_vars))
    #     print(set(model_vars) - set(base_model_vars))
    #     self.assertFalse(set(base_model_vars) - set(model_vars))
    #     self.assertFalse({v for v in set(model_vars) - set(base_model_vars) if 'pos' not in v})

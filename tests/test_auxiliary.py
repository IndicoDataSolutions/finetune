import os
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import unittest
import spacy
import random
import json
import copy
import shutil

from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from finetune import Classifier, SequenceLabeler
from finetune.base_models import TextCNN, BERTModelCased, GPT2Model, GPTModel, RoBERTa, GPT
from finetune.config import get_config
from finetune.util.metrics import (
    sequence_labeling_token_precision,
    sequence_labeling_token_recall,
)
from finetune.datasets.reuters import Reuters
from finetune.encoding.input_encoder import get_default_context, tokenize_context, ArrayEncodedOutput


# prevent excessive warning logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class TestAuxiliaryTokenization(unittest.TestCase):
    def test_get_default_context(self):
        context = [
            (2, ["single", True, 23.3, 4]),
            (4, ["double", True, 24.3, 2]),
            (8, ["single", False, 25.3, 3]),
        ]

        expected = ["single", True, 24.3, 3]
        self.assertEqual(get_default_context(context), expected)

    def test_tokenize_context(self):
        encoded_output = ArrayEncodedOutput(
            token_ids=[
                [40478, 40481],
                [ 1180, 40482],
                [  535, 40483],
                [  808, 40484],
                [  289, 40485],
                [17164, 40486],
                [40480, 40487]
            ],
            tokens=[40478, 'everything</w>', "'s</w>", 'only</w>', '$</w>', '80</w>', 40480],
            labels=[0] * 7,
            char_locs=[-1, 10, 12, 17, 19, 21, -1],
            mask=[0, 1, 1, 1, 1, 1, 0],
        )
        context = [
            {'token': "everything's", 'start': 0, 'end': 12, 'left': 10, 'bold': False},
            {'token': "only", 'start': 13, 'end': 17, 'left': 20, 'bold': False},
            {'token': "$80", 'start': 18, 'end': 21, 'left': 30, 'bold': True},
        ]
        expanded_context = tokenize_context(context, encoded_output)
        expected = [
            [False, 20],
            [False, 10],
            [False, 10],
            [False, 20],
            [True, 30],
            [True, 30],
            [False, 20]
        ]
        print(expanded_context)
        np.testing.assert_array_equal(expected, expanded_context)

class TestAuxiliary(unittest.TestCase):
    base_model = GPT
    default_context = {"token": "", "pos": "filler", "capitalized": "False"}

    @classmethod
    def setUpClass(self):
        self.trainX = ['i like apples'] * 2
        self.trainY = ['A', 'B']
        # labels could only be inferred given context
        self.trainY_seq = [
            [
                {'start': 0, 'end': 1, 'label': 'IMPORTANT', 'text': 'i'},
            ],
            [
                {'start': 7, 'end': 13, 'label': 'IMPORTANT', 'text': 'apples'},
            ]
        ]
        self.train_context = [
            [
                {'token': 'i', 'start': 0, 'end': 1, 'bold': True},
                {'token': 'like', 'start': 2, 'end': 6, 'bold': False},
                {'token': 'apples', 'start': 7, 'end': 13, 'bold': False},
            ],
            [
                {'token': 'i', 'start': 0, 'end': 1,  'bold': False},
                {'token': 'like', 'start': 2, 'end': 6,  'bold': False},
                {'token': 'apples', 'start': 7, 'end': 13,  'bold': True},
            ]
        ]

    def setUp(self):
        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn(
                "tests/saved-models still exists, it is possible that some test is not cleaning up properly."
            )
            pass

    def tearDown(self):
        shutil.rmtree("tests/saved-models/")

    def default_config(self, **kwargs):
        defaults = {
            "batch_size": 2,
            "max_length": 256,
            "n_epochs": 1000,
            "base_model": self.base_model,
            "val_size": 0,
            "use_auxiliary_info": True,
            "context_dim": 1,
            "val_set": (self.trainX, self.trainY, self.train_context)
        }
        defaults.update(kwargs)
        return dict(get_config(**defaults))
    
    def test_classifier_auxiliary(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        model = Classifier(
            **self.default_config()
        )
        model.fit(self.trainX, self.trainY, context=self.train_context)
        _ = model.predict(self.trainX, context=self.train_context)
        # test cached predict
        _ = model.predict(self.trainX, context=self.train_context)

    def test_classifier_no_auxiliary(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        config = self.default_config(use_auxiliary_info=False, context_dim=None, val_set=(self.trainX, self.trainY))
        model = Classifier(
            **config
        )
        model.fit(self.trainX, self.trainY)
        _ = model.predict(self.trainX)
        # test cached predict
        _ = model.predict(self.trainX)
    
    def _evaluate_sequence_preds(self, preds, includes_context):
        token_precision = sequence_labeling_token_precision(self.trainY_seq, preds)
        token_recall = sequence_labeling_token_recall(self.trainY_seq, preds)
        self.assertIn("IMPORTANT", token_precision)
        self.assertIn("IMPORTANT", token_recall)
        token_precision = np.mean(list(token_precision.values()))
        token_recall = np.mean(list(token_recall.values()))
        print(token_precision)
        print(token_recall)
        if includes_context:
            self.assertEqual(token_precision, 1.0)
            self.assertEqual(token_recall, 1.0)
        else:
            self.assertLessEqual(token_precision, 1.0)
            self.assertLessEqual(token_recall, 1.0)


    def test_sequence_labeler_no_auxiliary(self):
        """
        Ensure model training does not error out
        Ensure model returns reasonable predictions
        """
        
        model = SequenceLabeler(**self.default_config(use_auxiliary_info=False, val_set=(self.trainX, self.trainY)))
        model.fit(self.trainX, self.trainY_seq)
        preds = model.predict(self.trainX)
        self._evaluate_sequence_preds(preds, False)
        

    def test_sequence_labeler_auxiliary(self):
        """
        Ensure model training does not error out
        Ensure model returns reasonable predictions
        """
        
        model = SequenceLabeler(**self.default_config())
        model.fit(self.trainX, self.trainY_seq, context=self.train_context)
        preds = model.predict(self.trainX, context=self.train_context)
        self._evaluate_sequence_preds(preds, True)
    
    def test_save_load(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file = "tests/saved-models/test-save-load"
        config = self.default_config(save_adam_vars=False, n_epochs=1)
        model = Classifier(**config)

        model.fit(self.trainX, self.trainY, context=self.train_context)
        predictions = model.predict(self.trainX, context=self.train_context)
        model.save(save_file)

        model = Classifier.load(save_file)
        new_predictions = model.predict(self.trainX, context=self.train_context)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])
    
class TestAuxiliaryBert(TestAuxiliary):
    base_model = BERTModelCased


class TestAuxiliaryGPT(TestAuxiliary):
    base_model = GPTModel


class TestAuxiliaryGPT2(TestAuxiliary):
    base_model = GPT2Model


class TestAuxiliaryRoberta(TestAuxiliary):
    base_model = RoBERTa

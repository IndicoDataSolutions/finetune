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
    def create_context(cls, nlp, text, default):
        to_search = text
        removed = 0
        doc = nlp(text)
        context = []
        for token in doc:
            token_context = copy.deepcopy(default)
            start = to_search.find(token.text)
            assert start != -1
            to_search = to_search[start + len(token.text) :]
            end = start + len(token.text)
            token_context = {}
            token_context["token"] = token.text
            token_context["pos"] = token.pos_ if token.pos_ != "PROPN" else "NOUN"
            token_context["capitalized"] = (
                "False" if token.text.lower() == token.text else "True"
            )
            token_context["start"] = start + removed
            token_context["end"] = end + removed
            context.append(token_context)
            removed += end
        return context

    @classmethod
    def setUpClass(cls):
        nlp = spacy.load("en_core_web_sm")

        random.seed(42)
        np.random.seed(42)
        dataset = Reuters().dataframe
        dataset["annotations"] = [
            json.loads(annotation) for annotation in dataset["annotations"]
        ]
        trainX, testX, trainY, testY = train_test_split(
            dataset.texts.values,
            dataset.annotations.values,
            test_size=0.3,
            random_state=42,
        )

        train_context, test_context = [], []
        for text in trainX:
            context = cls.create_context(nlp, text, cls.default_context)
            train_context.append(context)

        for text in testX:
            context = cls.create_context(nlp, text, cls.default_context)
            test_context.append(context)

        cls.dataset = (trainX, testX, trainY, testY, train_context, test_context)

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
            "n_epochs": 1,
            "base_model": self.base_model,
            "val_size": 0,
            "use_auxiliary_info": True
            # "default_context": self.default_context
        }
        defaults.update(kwargs)
        return dict(get_config(**defaults))
    
    def test_simple_auxiliary(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        trainX = ['i like apples'] * 2
        trainY = ['q', 'w']
        train_context = [
            [
                {'token': 'i', 'start': 0, 'end': 1, 'pos_x':2, 'pos_y': 3},
                {'token': 'like', 'start': 2, 'end': 6, 'pos_x':3, 'pos_y': 3},
                {'token': 'i', 'start': 7, 'end': 13, 'pos_x':4, 'pos_y': 3},
            ],
            [
                {'token': 'i', 'start': 0, 'end': 1, 'pos_x':2, 'pos_y': 10},
                {'token': 'like', 'start': 2, 'end': 6, 'pos_x':3, 'pos_y': 10},
                {'token': 'i', 'start': 7, 'end': 13, 'pos_x':4, 'pos_y': 10},
            ]
        ]
        model = Classifier(
            **self.default_config()
        )
        model.fit(trainX, trainY, context=train_context)
        _ = model.predict(["everything's only $80"])
        _ = model.predict(trainX, context=train_context)
        # test cached predict
        _ = model.predict(trainX, context=train_context)

    def test_simple_no_auxiliary(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        trainX = ['i like apples'] * 2
        trainY = ['q', 'w']
        train_context = [
            [
                {'token': 'i', 'start': 0, 'end': 1, 'pos_x':2, 'pos_y': 3},
                {'token': 'like', 'start': 2, 'end': 6, 'pos_x':3, 'pos_y': 3},
                {'token': 'i', 'start': 7, 'end': 13, 'pos_x':4, 'pos_y': 3},
            ],
            [
                {'token': 'i', 'start': 0, 'end': 1, 'pos_x':2, 'pos_y': 10},
                {'token': 'like', 'start': 2, 'end': 6, 'pos_x':3, 'pos_y': 10},
                {'token': 'i', 'start': 7, 'end': 13, 'pos_x':4, 'pos_y': 10},
            ]
        ]
        config = self.default_config().copy()
        config.update({'use_auxiliary_info': False, 'context_dim': None})
        model = Classifier(
            **config
        )
        model.fit(trainX, trainY)
        _ = model.predict(trainX)
        # test cached predict
        _ = model.predict(trainX)

    def test_auxiliary_classifier(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        (trainX, testX, trainY, _, train_context, test_context) = self.dataset
        trainY = [
            random.randint(0, 1) for _ in range(len(trainY))
        ]  # random labels just to make sure there are no errors -> reasonable predictions tests are in sequence_label
        model = Classifier(
            **self.default_config()
        )
        model.fit(trainX, trainY, context=train_context)
        _ = model.predict(testX, context=test_context)
    
    def test_auxiliary_sequence_labeler(self):
        """
        Ensure model training does not error out
        Ensure model returns reasonable predictions
        """
        (trainX, testX, trainY, testY) = self.dataset
        model = SequenceLabeler(
            **self.default_config()
        )
        model.fit(trainX, trainY)
        preds = model.predict(testX)
        token_precision = sequence_labeling_token_precision(preds, testY)
        token_recall = sequence_labeling_token_recall(preds, testY)
        self.assertIn("Named Entity", token_precision)
        self.assertIn("Named Entity", token_recall)
        token_precision = np.mean(list(token_precision.values()))
        token_recall = np.mean(list(token_recall.values()))
        self.assertGreater(token_precision, 0.6)
        self.assertGreater(token_recall, 0.6)

    
    def test_save_load(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file = "tests/saved-models/test-save-load"
        config = self.default_config(save_adam_vars=False, n_epochs=1)
        model = Classifier(**config)

        (trainX, testX, trainY, _) = self.dataset
        trainY = [random.randint(0, 1) for _ in range(len(trainY))]
        model.fit(trainX, trainY)
        predictions = model.predict(testX)
        model.save(save_file)

        model = Classifier.load(save_file)
        new_predictions = model.predict(testX)
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

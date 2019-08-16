import os
import unittest
import logging
import shutil
import random
import copy
import requests
import json
import pytest
import spacy
import warnings

# prevent excessive warning logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from finetune import Classifier, SequenceLabeler
from finetune.base_models import GPT
from finetune.config import get_config
from finetune.util.metrics import (
    sequence_labeling_token_precision,
    sequence_labeling_token_recall,
)
from finetune.datasets.reuters import Reuters


class TestAuxiliary(unittest.TestCase):
    default_context = {"token": "", "pos": "filler", "capitalized": "False"}
    base_model = GPT

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
        trainX = [trainX, train_context]

        for text in testX:
            context = cls.create_context(nlp, text, cls.default_context)
            test_context.append(context)
        testX = [testX, test_context]

        cls.dataset = (trainX, testX, trainY, testY)

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
            "n_epochs": 3,
            "base_model": self.base_model,
            "val_size": 0,
        }
        defaults.update(kwargs)
        return dict(get_config(**defaults))
    
    def test_auxiliary_classifier(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        (trainX, testX, trainY, _) = self.dataset
        trainY = [
            random.randint(0, 1) for _ in range(len(trainY))
        ]  # random labels just to make sure there are no errors -> reasonable predictions tests are in sequence_label
        model = Classifier(
            default_context=self.default_context, **self.default_config()
        )
        model.fit(trainX, trainY)
        _ = model.predict(testX)
    
    def test_auxiliary_sequence_labeler(self):
        """
        Ensure model training does not error out
        Ensure model returns reasonable predictions
        """
        (trainX, testX, trainY, testY) = self.dataset
        model = SequenceLabeler(
            default_context=self.default_context, **self.default_config()
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
        model = Classifier(default_context=self.default_context, **config)

        (trainX, testX, trainY, _) = self.dataset
        trainY = [random.randint(0, 1) for _ in range(len(trainY))]
        model.fit(trainX, trainY)
        predictions = model.predict(testX)
        model.save(save_file)

        model = Classifier.load(save_file)
        new_predictions = model.predict(testX)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])
    

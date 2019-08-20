import os
import unittest
import logging
import shutil
import string
import gc
from copy import copy
import time
from pathlib import Path
from unittest.mock import MagicMock
import warnings

# prevent excessive warning logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

from finetune import LogisticRegressionClassifier
from finetune.model import PredictMode
from finetune.base_models import GPTModelSmall
from finetune.datasets import generic_download
from finetune.config import get_config
from finetune.errors import FinetuneError

SST_FILENAME = "SST-binary.csv"

SKIP_LM_TESTS = get_config().base_model.is_bidirectional


class TestLogisticRegressionClassifier(unittest.TestCase):
    n_sample = 20
    dataset_path = os.path.join("Data", "Classify", "SST-binary.csv")

    @classmethod
    def _download_sst(cls):
        """
        Download Stanford Sentiment Treebank to data directory
        """
        path = Path(cls.dataset_path)
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME,
        )

    @classmethod
    def setUpClass(cls):
        cls._download_sst()

    def setUp(self):
        self.dataset = pd.read_csv(self.dataset_path, nrows=self.n_sample * 3)
        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn(
                "tests/saved-models still exists, it is possible that some test is not cleaning up properly."
            )
            pass

    def tearDown(self):
        shutil.rmtree("tests/saved-models/")

    def test_multiple_models_fit_predict(self):
        """
        Ensure second call to predict is faster than first
        """
        model = LogisticRegressionClassifier()
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text.values, train_sample.Target.values)
        model.predict(valid_sample.Text.values)

        model2 = LogisticRegressionClassifier()
        model2.fit(train_sample.Text.values, train_sample.Target.values)
        model2.predict(valid_sample.Text.values)


    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = LogisticRegressionClassifier()
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)

        with self.assertRaises(FinetuneError):
            model.fit(train_sample.Text, train_sample.Target[:1])

        model.fit(train_sample.Text.values, train_sample.Target.values)

        predictions = model.predict(valid_sample.Text.values)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

        probabilities = model.predict_proba(valid_sample.Text.values)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_fit_predict_tfidf(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = LogisticRegressionClassifier(encoder='tfidf')
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)

        with self.assertRaises(FinetuneError):
            model.fit(train_sample.Text, train_sample.Target[:1])

        model.fit(train_sample.Text.values, train_sample.Target.values)

        predictions = model.predict(valid_sample.Text.values)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

        probabilities = model.predict_proba(valid_sample.Text.values)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_save_load(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file = "tests/saved-models/test-save-load"
        model = LogisticRegressionClassifier()
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        predictions = model.predict(valid_sample.Text)

        # testing file size reduction options
        model.save(save_file)
        self.assertLess(os.stat(save_file).st_size, 500000000)

        model = LogisticRegressionClassifier.load(save_file)
        new_predictions = model.predict(valid_sample.Text)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])

    def test_reasonable_predictions(self):
        """
        Ensure model converges to a reasonable solution for a trivial problem
        """
        model = LogisticRegressionClassifier()
        n_per_class = self.n_sample * 5
        trX = ["cat"] * n_per_class + ["finance"] * n_per_class
        trY = copy(trX)
        teX = ["feline"] + ["investment"]
        teY = ["cat"] + ["finance"]
        model.fit(trX, trY)
        predY = model.predict(teX)
        self.assertEqual(accuracy_score(teY, predY), 1.00)
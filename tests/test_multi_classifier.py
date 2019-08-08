import os
import unittest
import logging
import shutil
import string
from pathlib import Path
from unittest.mock import MagicMock
import warnings

# prevent excessive warning logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import pandas as pd
import numpy as np

from finetune import MultiLabelClassifier
from finetune.datasets import generic_download
from finetune.config import get_config

SST_FILENAME = "SST-binary.csv"


class TestMultiLabelClassifier(unittest.TestCase):
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

    def default_config(self, **kwargs):
        return dict(
            batch_size=2,
            max_length=128,
            n_epochs=5,
            l2_reg=0.0,
            clf_p_drop=0.0,
            **kwargs
        )

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = MultiLabelClassifier(**self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, [[t, 6, 3] for t in train_sample.Target])

        probabilities = model.predict_proba(valid_sample.Text)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

        predictions = model.predict(valid_sample.Text)
        for prediction in predictions:
            self.assertIsInstance(prediction[0], (str, np.int, np.int64))
            self.assertIn(3, prediction)
            self.assertIn(6, prediction)


import os
import unittest
import warnings
from pathlib import Path

# prevent excessive warning logs 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np

from finetune import MultifieldClassifier, MultifieldRegressor, Regressor
from finetune.config import get_config
from finetune.datasets import generic_download

SST_FILENAME = "SST-binary.csv"


class TestModel(unittest.TestCase):
    n_sample = 100
    n_hidden = 768
    dataset_path = os.path.join(
        'Data', 'Classify', SST_FILENAME
    )

    @classmethod
    def _download_sst(cls):
        """
        Download Stanford Sentiment Treebank to data directory
        """
        path = Path(cls.dataset_path)
        if path.exists():
            return

        path.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME
        )

    @classmethod
    def setUpClass(cls):
        cls._download_sst()

    def setUp(self):
        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn("tests/saved-models still exists, it is possible that some test is not cleaning up properly.")
            pass
        self.save_file = 'tests/saved-models/test-save-load.jl'
        self.dataset = pd.read_csv(self.dataset_path)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        self.text_data_train = [[x] * 3 for x in train_sample.Text.values.tolist()]
        self.text_data_valid = [[x] * 3 for x in valid_sample.Text.values.tolist()]
        self.train_targets = train_sample.Target
        tf.reset_default_graph()

    def test_multifield_classify(self):
        """
        Ensure fit predict works on classification with multi inputs
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        self.model = MultifieldClassifier()
        self.model.fit(self.text_data_train, self.train_targets)
        predictions = self.model.predict(self.text_data_valid)
        self.model.save(self.save_file)
        model = MultifieldRegressor.load(self.save_file)
        new_predictions = model.predict(self.text_data_valid)
        for new_pred, old_pred in zip(new_predictions, predictions):
            self.assertEqual(new_pred, old_pred)

    def test_multifield_regression(self):
        """                                                                                                                                                                         
        Ensure fit predict works with regression targets and multiple inputs.
        Ensure saving + loading does not cause errors                                                                                                                               
        Ensure saving + loading does not change predictions                                                                                                                         
        """
        self.model = MultifieldRegressor()
        self.model.fit(self.text_data_train, [np.random.random() for _ in self.train_targets])
        predictions = self.model.predict(self.text_data_valid)
        self.model.save(self.save_file)
        model = MultifieldRegressor.load(self.save_file)
        new_predictions = model.predict(self.text_data_valid)
        for new_pred, old_pred in zip(new_predictions, predictions):
            self.assertAlmostEqual(new_pred, old_pred, places=2)

    def test_regressor(self):
        n_samples = 20
        x_test = np.array(['the quick fox jumped over the lazy brown dog'] * n_samples)
        y_test = np.random.random(n_samples)
        model_test = Regressor(n_epochs=1)
        model_test.fit(x_test, y_test)
        preds = model_test.predict(x_test)
        self.assertIsInstance(preds, list)
        self.assertIsInstance(preds[0], float)

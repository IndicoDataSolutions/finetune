import os
import unittest
import warnings
from pathlib import Path

# prevent excessive warning logs 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import enso
from enso.download import generic_download

from finetune import Model
from finetune.config import get_hparams
import numpy as np

SST_FILENAME = "SST-binary.csv"


class TestModel(unittest.TestCase):
    n_sample = 100
    n_hidden = 768
    dataset_path = os.path.join(
        enso.config.DATA_DIRECTORY, 'Classify', 'SST-binary.csv'
    )

    @classmethod
    def _download_sst(cls):
        """
        Download Stanford Sentiment Treebank to enso `data` directory
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
        save_file_autosave = 'tests/saved-models/autosave_path'
        self.save_file = 'tests/saved-models/test-save-load'
        hparams = get_hparams(batch_size=2, max_length=256)
        self.model = Model(hparams=hparams, verbose=False, autosave_path=save_file_autosave)

        self.dataset = pd.read_csv(self.dataset_path)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        self.text_data_train = list(zip(train_sample.Text, train_sample.Text, train_sample.Text))
        self.text_data_valid = list(zip(valid_sample.Text, valid_sample.Text, valid_sample.Text))
        self.train_targets = train_sample.Target
        tf.reset_default_graph()

    def test_multifield_classify(self):
        """
        Ensure fit predict works on classification with multi inputs
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        self.model.fit(self.text_data_train, self.train_targets)
        self.assertTrue(self.model.is_classification)
        predictions = self.model.predict(self.text_data_valid)
        self.model.save(self.save_file)
        model = Model.load(self.save_file)
        new_predictions = model.predict(self.text_data_valid)
        for new_pred, old_pred in zip(new_predictions, predictions):
            self.assertEqual(new_pred, old_pred)

    def test_multifield_regression(self):
        """                                                                                                                                                                         
        Ensure fit predict works with regression targets and multiple inputs.
        Ensure saving + loading does not cause errors                                                                                                                               
        Ensure saving + loading does not change predictions                                                                                                                         
        """
        self.model.fit(self.text_data_train, [np.random.random() for _ in self.train_targets])
        self.assertTrue(not self.model.is_classification)
        predictions = self.model.predict(self.text_data_valid)
        self.model.save(self.save_file)
        model = Model.load(self.save_file)
        new_predictions = model.predict(self.text_data_valid)
        for new_pred, old_pred in zip(new_predictions, predictions):
            self.assertEqual(new_pred, old_pred)

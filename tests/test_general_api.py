import os
import unittest

from pathlib import Path

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import enso
from enso.download import generic_download

from finetune import LanguageModelGeneralAPI

SST_FILENAME = "SST-binary.csv"


class TestLanguageModelClassifier(unittest.TestCase):

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
        self.dataset = pd.read_csv(self.dataset_path)
        tf.reset_default_graph()

    def test_multifield_classify(self):
        """
        Ensure fit predict works.
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file_autosave = 'tests/saved-models/autosave_path'
        save_file = 'tests/saved-models/test-save-load'
        model = LanguageModelGeneralAPI(verbose=False, autosave_path=save_file_autosave)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit([train_sample.Text] * 3, train_sample.Target)
        self.assertTrue(model.is_classification)
        predictions = model.predict([valid_sample.Text] * 3)
        model.save(save_file)
        model = LanguageModelGeneralAPI.load(save_file)
        new_predictions = model.predict([valid_sample.Text] * 3)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])


import os
import unittest
import logging
import shutil
import string
from copy import copy
from pathlib import Path
from unittest.mock import MagicMock
import warnings

# prevent excessive warning logs 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from finetune import Classifier
from finetune.datasets import generic_download
from finetune.config import get_config

SST_FILENAME = "SST-binary.csv"


class TestClassifier(unittest.TestCase):
    n_sample = 20
    n_hidden = 768
    dataset_path = os.path.join(
        'Data', 'Classify', 'SST-binary.csv'
    )

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
            filename=SST_FILENAME
        )

    @classmethod
    def setUpClass(cls):
        cls._download_sst()

    def setUp(self):
        self.dataset = pd.read_csv(self.dataset_path, nrows=self.n_sample * 3)
        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn("tests/saved-models still exists, it is possible that some test is not cleaning up properly.")
            pass

        tf.reset_default_graph()

    def tearDown(self):
        shutil.rmtree("tests/saved-models/")

    def default_config(self, **kwargs):
        return get_config(
            batch_size=2,
            max_length=128,
            n_epochs=1,
            verbose=False,
            **kwargs
        )

    def test_fit_lm_only(self):
        """
        Ensure LM only training does not error out
        """
        model = Classifier()
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)

        # Ensure model can still be fit with only text
        model.fit(train_sample.Text)

        # Ensure model can still be fit with text + targets
        model.fit(train_sample.Text, train_sample.Target)

        predictions = model.predict(valid_sample.Text)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

        probabilities = model.predict_proba(valid_sample.Text)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Classifier(config=self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)

        predictions = model.predict(valid_sample.Text)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

        probabilities = model.predict_proba(valid_sample.Text)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_fit_predict_batch_size_1(self):
        """
        Ensure training is possible with batch size of 1
        """
        model = Classifier(config=self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)

    def test_save_load(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file = 'tests/saved-models/test-save-load'
        model = Classifier(config=self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        predictions = model.predict(valid_sample.Text)
        model.save(save_file)
        model = Classifier.load(save_file)
        new_predictions = model.predict(valid_sample.Text)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])

    def test_featurize(self):
        """
        Ensure featurization returns an array of the right shape
        Ensure featurization is still possible after fit
        """
        model = Classifier(config=self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        features = model.featurize(train_sample.Text)
        self.assertEqual(features.shape, (self.n_sample, self.n_hidden))
        model.fit(train_sample.Text, train_sample.Target)
        features = model.featurize(train_sample.Text)
        self.assertEqual(features.shape, (self.n_sample, self.n_hidden))

    def test_reasonable_predictions(self):
        """
        Ensure model converges to a reasonable solution for a trivial problem
        """
        model = Classifier(config=self.default_config())
        n_per_class = (self.n_sample * 5)
        trX = ['cat'] * n_per_class + ['finance'] * n_per_class
        trY = copy(trX)
        teX = ['feline'] * n_per_class + ['investment'] * n_per_class
        teY = ['cat'] * n_per_class + ['finance'] * n_per_class
        model.fit(trX, trY)
        predY = model.predict(teX)
        self.assertEqual(accuracy_score(teY, predY), 1.00)

    def test_language_model(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        model = Classifier(verbose=False)
        lm_out = model.generate_text("", max_length=5)
        self.assertEqual(type(lm_out), str)
        lm_out_2 = model.generate_text("Indico RULE")
        self.assertEqual(type(lm_out_2), str)
        self.assertIn('_start_Indico RULE'.lower(), lm_out_2)

    def test_save_load_language_model(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file = 'tests/saved-models/test-save-load'
        model = Classifier(verbose=False)
        train_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        lm_out = model.generate_text("", 5)
        self.assertEqual(type(lm_out), str)
        model.save(save_file)
        model = Classifier.load(save_file)
        lm_out_2 = model.generate_text("Indico RULE")
        self.assertEqual(type(lm_out_2), str)
        self.assertIn('_start_Indico RULE'.lower(), lm_out_2)

<<<<<<< HEAD
    def test_early_termination_lm(self):
        model = Classifier(verbose=False)

        # A dirty mock to make all model inferences output a hundred _classify_ tokens
        def mock_init_from_pretrained(*args, **kwargs):
            model.sess = MagicMock()
            model.sess.run = MagicMock(return_value=100 * [model.encoder['_classify_']])

        model._init_from_pretrained = mock_init_from_pretrained
        lm_out = model.generate_text()
        self.assertEqual(lm_out, '_start__classify_')

=======
>>>>>>> c1e6ea5... WIP: refactor target_type logic and subclasses
    def test_validation(self):
        """
        Ensure validation settings do not result in an error
        """
        config = self.default_config(val_interval=10, val_size=0.5)
        model = Classifier(config=config)
        train_sample = self.dataset.sample(n=20)
        model.fit(train_sample.Text, train_sample.Target)

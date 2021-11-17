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

from finetune import Classifier
from finetune.model import PredictMode
from finetune.base_models import GPTModelSmall, GPT
from finetune.datasets import generic_download
from finetune.config import get_config
from finetune.errors import FinetuneError

SST_FILENAME = "SST-binary.csv"

class TestClassifier(unittest.TestCase):
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
        defaults = {"batch_size": 2, "max_length": 128, "n_epochs": 1}
        defaults.update(kwargs)
        return dict(get_config(**defaults))

    def test_fit_lm_only(self):
        """
        Ensure LM only training does not error out
        """
        model = Classifier(base_model=GPT)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)

        # Ensure model can still be fit with only text
        model.fit(train_sample.Text)

        # Save and reload check
        save_file = "tests/saved-models/test-save-load"
        model.save(save_file)
        model = Classifier.load(save_file)

        # Ensure model can still be fit with text + targets
        model.fit(train_sample.Text, train_sample.Target)
        predictions = model.predict(valid_sample.Text)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

        probabilities = model.predict_proba(valid_sample.Text)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_multiple_models_fit_predict(self):
        """
        Ensure second call to predict is faster than first
        """
        model = Classifier(**self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text.values, train_sample.Target.values)
        model.predict(valid_sample.Text.values)

        model2 = Classifier(**self.default_config())
        model2.fit(train_sample.Text.values, train_sample.Target.values)
        model2.predict(valid_sample.Text.values)

    def test_cached_predict(self):
        """
        Ensure second call to predict is faster than first
        """

        model = Classifier(**self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text.values, train_sample.Target.values)

        with model.cached_predict():
            start = time.time()
            model.predict(valid_sample.Text[:1].values)
            first = time.time()
            model.predict(valid_sample.Text[:1].values)
            second = time.time()

        first_prediction_time = first - start
        second_prediction_time = second - first
        self.assertLess(second_prediction_time, first_prediction_time / 2.0)

    def test_correct_cached_predict(self):
        model = Classifier(**self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)

        # Test with different sizes to make sure we handle cases where
        # the data doesn't divide evenly into batches
        half_sample = int(self.n_sample / 2)
        quarter_sample = int(half_sample / 2)

        model.fit(train_sample.Text.values, train_sample.Target.values)

        # Predictions w/o cached predict
        preds = [
            model.predict_proba(valid_sample.Text.values[:half_sample]),
            model.predict_proba(valid_sample.Text.values[half_sample:]),
            model.predict_proba(valid_sample.Text.values[:quarter_sample]),
            model.predict_proba(valid_sample.Text.values[quarter_sample:]),
        ]

        # Predictions w/ cached predict
        with model.cached_predict():
            cached_preds = [
                model.predict_proba(valid_sample.Text.values[:half_sample]),
                model.predict_proba(valid_sample.Text.values[half_sample:]),
                model.predict_proba(valid_sample.Text.values[:quarter_sample]),
                model.predict_proba(valid_sample.Text.values[quarter_sample:]),
            ]

        for batch_preds, batch_cached_preds in zip(preds, cached_preds):
            for pred, cached_pred in zip(batch_preds, batch_cached_preds):
                assert list(pred.keys()) == list(cached_pred.keys())
                for pred_val, cached_pred_val in zip(
                    pred.values(), cached_pred.values()
                ):
                    np.testing.assert_almost_equal(pred_val, cached_pred_val, decimal=4)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Classifier(**self.default_config())
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

    def test_fit_predict_low_memory(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Classifier(**self.default_config(low_memory_mode=True))
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)

        model.fit(train_sample.Text.values, train_sample.Target.values)

        predictions = model.predict(valid_sample.Text.values)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

        probabilities = model.predict_proba(valid_sample.Text.values)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_class_weights(self):
        # testing class weights
        train_sample = self.dataset.sample(n=self.n_sample * 3)
        valid_sample = self.dataset.sample(n=self.n_sample * 3)
        model = Classifier(**self.default_config())
        model.fit(train_sample.Text.values, train_sample.Target.values)
        predictions = model.predict(valid_sample.Text.values)
        recall = recall_score(valid_sample.Target.values, predictions, pos_label=1)
        model = Classifier(**self.default_config(class_weights={1: 100}))
        model.fit(train_sample.Text.values, train_sample.Target.values)
        predictions = model.predict(valid_sample.Text.values)
        new_recall = recall_score(valid_sample.Target.values, predictions, pos_label=1)
        self.assertTrue(new_recall >= recall)

        # test auto-inferred class weights function
        model = Classifier(**self.default_config(class_weights="log"))
        model.fit(train_sample.Text.values, train_sample.Target.values)

    def test_chunk_long_sequences(self):
        test_sequence = [
            "This is a sentence to test chunk_long_sequences in classification. " * 20,
            "Another example so now there are two different classes in the test. " * 20,
        ]
        labels = ["a", "b"]
        model = Classifier()
        model.config.chunk_long_sequences = True
        model.config.max_length = 18

        model.finetune(test_sequence * 10, labels * 10)

        predictions = model.predict(test_sequence * 10)
        probas = model.predict_proba(test_sequence * 10)

        self.assertEqual(len(predictions), 20)
        self.assertEqual(len(probas[0]), 2)
        np.testing.assert_almost_equal(np.sum(list(probas[0].values())), 1, decimal=4)

    def test_fit_predict_batch_size_1(self):
        """
        Ensure training is possible with batch size of 1
        """
        model = Classifier(**self.default_config())
        model.config.batch_size = 1
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text.values, train_sample.Target.values)
        model.predict(valid_sample.Text.values)

    def test_save_load(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file = "tests/saved-models/test-save-load"
        config = self.default_config(save_adam_vars=False)
        model = Classifier(**config)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        predictions = model.predict(valid_sample.Text)

        # testing file size reduction options
        model.save(save_file)
        self.assertLess(os.stat(save_file).st_size, 500000000)

        # reducing floating point precision
        model.saver.save_dtype = np.float16
        model.save(save_file)
        self.assertLess(os.stat(save_file).st_size, 251000000)

        model = Classifier.load(save_file)
        new_predictions = model.predict(valid_sample.Text)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])

    def test_featurize(self):
        """
        Ensure featurization returns an array of the right shape
        Ensure featurization is still possible after fit
        """
        model = Classifier(**self.default_config())
        train_sample = self.dataset.sample(n=self.n_sample)
        features = model.featurize(train_sample.Text)
        self.assertEqual(features.shape, (self.n_sample, model.config.n_embed))
        model.fit(train_sample.Text, train_sample.Target)
        features = model.featurize(train_sample.Text)
        self.assertEqual(features.shape, (self.n_sample, model.config.n_embed))

    def test_reasonable_predictions(self):
        """
        Ensure model converges to a reasonable solution for a trivial problem
        """
        model = Classifier(**self.default_config(n_epochs=5))
        
        n_duplicates = 5

        trX = (
            ["cat", "kitten", "feline", "meow", "kitty"] * n_duplicates + 
            ["finance", "investment", "investing", "dividends", "financial"] * n_duplicates
        )
        trY = (
            ['cat'] * (len(trX) // 2) + ['finance'] * (len(trX) // 2)
        )
        teX = ["furball", "fiduciary"]
        teY = ["cat"] + ["finance"]
        model.fit(trX, trY)
        predY = model.predict(teX)
        print(predY)
        self.assertEqual(accuracy_score(teY, predY), 1.00)

    def test_reasonable_predictions_smaller_model(self):
        """
        Ensure model converges to a reasonable solution for a trivial problem
        """
        model = Classifier(base_model=GPTModelSmall)
        n_per_class = self.n_sample * 5
        trX = ["cat"] * n_per_class + ["finance"] * n_per_class
        np.random.shuffle(trX)
        trY = copy(trX)
        teX = ["feline"] * n_per_class + ["investment"] * n_per_class
        teY = ["cat"] * n_per_class + ["finance"] * n_per_class
        model.fit(trX, trY)
        predY = model.predict(teX)
        self.assertEqual(accuracy_score(teY, predY), 1.00)

    def test_save_load_language_model(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file = "tests/saved-models/test-save-load"
        model = Classifier(base_model=GPT)

        lm_out = model.generate_text("The quick brown fox", 6)
        start_id = model.input_pipeline.text_encoder.start_token
        start_token = model.input_pipeline.text_encoder.decoder[start_id]
        self.assertNotIn(start_token, lm_out) # Non finetuned models do not use extra tokens
        
        train_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        lm_out = model.generate_text("", 5)
        self.assertIn(start_token, lm_out.lower())
        self.assertEqual(type(lm_out), str)
        model.save(save_file)

        model = Classifier.load(save_file)
        lm_out_2 = model.generate_text("Indico RULE")
        self.assertEqual(type(lm_out_2), str)
        
        self.assertIn("{}Indico RULE".format(start_token).lower(), lm_out_2.lower()) # Both of these models use extra toks

    def test_generate_text_stop_early(self):
        model = Classifier(base_model=GPT)

        # A dirty mock to make all model inferences output a hundred _classify_ tokens
        fake_estimator = MagicMock()
        model.get_estimator = lambda *args, **kwargs: (fake_estimator, [])
        model.input_pipeline.text_encoder._lazy_init()
        fake_estimator.predict = MagicMock(
            return_value=iter(
                [{PredictMode.GENERATE_TEXT: 100 * [model.input_pipeline.text_encoder["_classify_"]]}]
            )
        )
        start_id = model.input_pipeline.text_encoder.start_token
        start_token = model.input_pipeline.text_encoder.decoder[start_id]
        lm_out = model.generate_text(use_extra_toks=True)
        self.assertEqual(lm_out, "{}_classify_".format(start_token))

    def test_validation(self):
        """
        Ensure validation settings do not result in an error
        """
        config = self.default_config(val_interval=10, val_size=10)
        model = Classifier(**config)
        train_sample = self.dataset.sample(n=20)
        model.fit(train_sample.Text, train_sample.Target)

    def test_fit_with_eval_acc(self):
        """
        Test issue #263
        """

        model = Classifier(**self.default_config(batch_size=3, eval_acc=True))
        train_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)

    def test_explain(self):
        model = Classifier(**self.default_config(base_model=GPT))
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        explanations = model.explain(valid_sample.Text)
        normal_predictions = model.predict(valid_sample.Text)
        explanation_preds = [e["prediction"] for e in explanations]

        # check that the process of turning on explain does not change the preds
        self.assertEqual(explanation_preds, list(normal_predictions))
        self.assertEqual(len(explanation_preds), len(train_sample.Text))
        self.assertEqual(type(explanations[0]["token_ends"]), list)
        self.assertEqual(type(explanations[0]["token_starts"]), list)
        self.assertEqual(type(explanations[0]["explanation"]), dict)
        self.assertEqual(
            len(explanations[0]["token_starts"]), len(explanations[0]["explanation"][0])
        )
        self.assertEqual(
            len(explanations[0]["token_ends"]), len(explanations[0]["explanation"][0])
        )


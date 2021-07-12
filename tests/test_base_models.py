import os
import shutil
import time
import warnings
import unittest
from pathlib import Path
import codecs
import json
import random

# prevent excessive warning logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from finetune.base_models import (
    TextCNN,
    FastTextCNN,
    BERTModelCased,
    GPT2Model,
    GPTModel,
    RoBERTa,
    OSCAR,
    TCNModel,
    DistilBERT,
)

from finetune import Classifier, Comparison, SequenceLabeler
from finetune.datasets import generic_download
from finetune.config import get_config
from finetune.errors import FinetuneError
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.util.metrics import (
    sequence_labeling_token_precision,
    sequence_labeling_token_recall,
    sequence_labeling_overlap_precision,
    sequence_labeling_overlap_recall,
)
try:
    from finetune.base_models.huggingface.models import HFBert, HFElectraGen, HFElectraDiscrim
    HUGGINGFACE_MODELS = True
except ImportError:
    HFBert = None
    HFElectraGen = None
    HFElectraDiscrim = None
    HUGGINGFACE_MODELS = False


SST_FILENAME = "SST-binary.csv"


class TestModelBase(unittest.TestCase):
    base_model = None
    model_specific_config = {}

    def default_config(cls, **kwargs):
        defaults = dict(
            base_model=cls.base_model,
            batch_size=2,
            max_length=128,
            lm_loss_coef=0.0,
            **cls.model_specific_config
        )
        defaults.update(kwargs)
        return defaults


class TestClassifierTextCNN(TestModelBase):
    n_sample = 20
    dataset_path = os.path.join("Data", "Classify", "SST-binary.csv")
    base_model = TextCNN

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
        try:
            shutil.rmtree("tests/saved-models/")
        except FileNotFoundError:
            pass

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
        model.fit(train_sample.Text.values, train_sample.Target.values)
        predictions = model.predict_proba(valid_sample.Text[:1].values)
        predictions2 = model.predict_proba(valid_sample.Text[1:2].values)
        with model.cached_predict():
            np.testing.assert_allclose(
                list(model.predict_proba(valid_sample.Text[:1].values)[0].values()),
                list(predictions[0].values()),
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                list(model.predict_proba(valid_sample.Text[1:2].values)[0].values()),
                list(predictions2[0].values()),
                rtol=1e-4,
            )

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

    def test_oversample(self):
        """
        Ensure model training does not error out when oversampling is set to True
        """

        model = Classifier(**self.default_config())
        model.config.oversample = True
        train_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text.values, train_sample.Target.values)

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
        save_file_fp16 = "tests/saved-models/test-save-load_fp16"

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
        model.save(save_file_fp16)
        self.assertLess(os.stat(save_file_fp16).st_size, 260000000)

        model = Classifier.load(save_file_fp16)
        new_predictions = model.predict(valid_sample.Text)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])

    def test_fp_16_equality(self):
        config = self.default_config(save_adam_vars=False, float_16_predict=True)
        model = Classifier(**config)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        predictions = model.predict(valid_sample.Text)
        model.config.float_16_predict = False
        # important to do it this way for the models that do not support fp16 as
        # config resolution is only done when the config is set correctly.
        new_predictions = model.predict(valid_sample.Text)
        errors = 0
        for prediction, new_prediction in zip(predictions, new_predictions):
            if prediction != new_prediction:
                errors += 1
        self.assertEqual(errors, 0)
            
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

    def test_featurize_sequence(self):
        _test_featurize_sequence(self, model_fn=Classifier)

    def test_validation(self):
        """
        Ensure validation settings do not result in an error
        """
        config = self.default_config()
        config.update({"val_interval": 10, "val_size": 0})
        model = Classifier(**config)
        train_sample = self.dataset.sample(n=20)
        model.fit(train_sample.Text, train_sample.Target)


class TestComparisonTextCNN(TestModelBase):
    n_sample = 20
    dataset_path = os.path.join("Data", "Classify", "SST-binary.csv")
    base_model = TextCNN

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Comparison(**self.default_config())
        n_samples = 10
        model.fit(
            [
                [
                    "Transformers was a terrible movie but a great model",
                    "Transformers are a great model but a terrible movie",
                ]
            ]
            * n_samples,
            ["yes"] * n_samples,
        )

        test_data = [
            [
                "Transformers was a terrible movie but a great model",
                "Transformers are a great model but a terrible movie",
            ]
        ]

        predictions = model.predict(test_data)
        for prediction in predictions:
            self.assertIsInstance(prediction, (str, bytes))

        probabilities = model.predict_proba(test_data)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)


class TestSequenceLabelerTextCNN(TestModelBase):
    n_sample = 100
    dataset_path = os.path.join("Data", "Sequence", "reuters.xml")
    processed_path = os.path.join("Data", "Sequence", "reuters.json")

    base_model = TextCNN

    @classmethod
    def _download_reuters(cls):
        """
        Download Reuters to test directory
        """
        path = Path(cls.dataset_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(cls.dataset_path):
            url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
            r = requests.get(url)
            with open(cls.dataset_path, "wb") as fp:
                fp.write(r.content)

        with codecs.open(cls.dataset_path, "r", "utf-8") as infile:
            soup = bs(infile, "html.parser")

        docs = []
        docs_labels = []
        for elem in soup.find_all("document"):
            texts = []
            labels = []

            # Loop through each child of the element under "textwithnamedentities"
            for c in elem.find("textwithnamedentities").children:
                if type(c) == Tag:
                    if c.name == "namedentityintext":
                        label = "Named Entity"  # part of a named entity
                    else:
                        label = "<PAD>"  # irrelevant word
                    texts.append(c.text)
                    labels.append(label)

            docs.append(texts)
            docs_labels.append(labels)

        with open(cls.processed_path, "wt") as fp:
            json.dump((docs, docs_labels), fp)

    @classmethod
    def setUpClass(cls):
        cls._download_reuters()

    def setUp(self):
        self.save_file = "tests/saved-models/test-save-load"
        random.seed(42)
        np.random.seed(42)
        with open(self.processed_path, "rt") as fp:
            self.texts, self.labels = json.load(fp)

        self.model = SequenceLabeler(**self.default_config())

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        Ensure class reweighting behaves as intended
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1
        )

        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)

        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]["confidence"], dict)

        token_precision = sequence_labeling_token_precision(
            test_annotations, predictions
        )
        token_recall = sequence_labeling_token_recall(test_annotations, predictions)
        overlap_precision = sequence_labeling_overlap_precision(
            test_annotations, predictions
        )
        overlap_recall = sequence_labeling_overlap_recall(test_annotations, predictions)

        self.assertIn("Named Entity", token_precision)
        self.assertIn("Named Entity", token_recall)
        self.assertIn("Named Entity", overlap_precision)
        self.assertIn("Named Entity", overlap_recall)

        self.model.save(self.save_file)

    def test_cached_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, _ = train_test_split(
            texts, annotations, test_size=0.1
        )
        self.model.fit(train_texts, train_annotations)
        with self.model.cached_predict():
            pred1 = self.model.predict(test_texts)
            pred2 = self.model.predict(test_texts)
        self.assertEqual(pred1, pred2)

    def test_fit_predict_multi_model(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        self.model = SequenceLabeler(
            **self.default_config(
                batch_size=2,
                max_length=256,
                lm_loss_coef=0.0,
                multi_label_sequences=True,
            )
        )
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, _ = train_test_split(
            texts, annotations, test_size=0.1
        )
        self.model.fit(train_texts, train_annotations)
        pre_save_preds = self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)
        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]["confidence"], dict)
        self.model.save(self.save_file)
        model = SequenceLabeler.load(self.save_file)
        post_load_preds = model.predict(test_texts)
        self.assertEqual(post_load_preds, pre_save_preds)

    def test_featurize_sequence(self):
        _test_featurize_sequence(self, model_fn=SequenceLabeler)


def _test_featurize_sequence(self, model_fn):
    test_input = ["""Pirateipsum: Corsair bilge rat interloper. Nipperkin
                  aye chase. Brigantine yard weigh anchor.  Jolly boat
                  American Main spirits. Letter of Marque reef sails cable.
                  Deadlights port nipper.  Fire ship gibbet American Main.
                  Jack Ketch cable fore.  Tack black jack draught.""",
                  """Nipperkin aye chase. Brigantine yard weigh anchor.
                  Jolly boat American Main spirits. Letter of Marque reef
                  sails cable.  Deadlights port nipper.  Fire ship gibbet
                  American Main.  Jack Ketch cable fore."""]
    non_chunk_config = self.default_config()
    non_chunk_config["chunk_long_sequences"] = False
    non_chunk_config["max_length"] = 256
    model = model_fn(**non_chunk_config)
    non_chunk_features = model.featurize_sequence(test_input)

    prev_features = [None] * len(test_input)
    for max_len in [32, 64, 128, 256]:
        chunk_config = self.default_config()
        chunk_config["chunk_long_sequences"] = True
        chunk_config["max_length"] = max_len
        model2 = model_fn(**chunk_config)
        chunk_features = model2.featurize_sequence(test_input)

        for non_chunk,chunk,prev,X in zip(non_chunk_features,
                                          chunk_features,
                                          prev_features,
                                          test_input):
            encoded = next(model.input_pipeline._text_to_ids(X))
            # subtract 2 to account for the start and end tokens
            encoded_len = len(encoded.token_ids) - 2
            self.assertTrue(non_chunk.shape[0] == encoded_len)
            self.assertTrue(chunk.shape[0] == encoded_len)
            self.assertTrue(non_chunk.shape == chunk.shape)
            if prev is not None:
                self.assertTrue(prev.shape == chunk.shape)
        prev_features = chunk_features


class TestSequenceLabelerFastTextCNN(TestSequenceLabelerTextCNN):
    base_model = FastTextCNN


class TestClassifierFastTextCNN(TestClassifierTextCNN):
    base_model = FastTextCNN


class TestComparisonFastTextCNN(TestComparisonTextCNN):
    base_model = FastTextCNN


class TestSequenceLabelerBert(TestSequenceLabelerTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = BERTModelCased


class TestClassifierBert(TestClassifierTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = BERTModelCased


class TestComparisonBert(TestComparisonTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = BERTModelCased


class TestSequenceLabelerRoberta(TestSequenceLabelerTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = RoBERTa


class TestClassifierRoberta(TestClassifierTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = RoBERTa


class TestComparisonRoberta(TestComparisonTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = RoBERTa


class TestClassifierTCN(TestClassifierTextCNN):
    base_model = TCNModel


class TestSequenceLabelerTCN(TestSequenceLabelerTextCNN):
    base_model = TCNModel


class TestClassifierDistilBERT(TestClassifierTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-2}
    base_model = DistilBERT


class TestSequenceLabelerDistilBERT(TestSequenceLabelerTextCNN):
    base_model = DistilBERT


class TestSequenceLabelerOscar(TestSequenceLabelerTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = OSCAR


class TestClassifierOscar(TestClassifierTextCNN):
    model_specific_config = {"n_epochs": 2, "lr": 1e-4}
    base_model = OSCAR


@unittest.skipIf(not HUGGINGFACE_MODELS, reason="Hugging face transformers could not be imported")
class TestSequenceHuggingfaceElectraGen(TestSequenceLabelerTCN):
    base_model = HFElectraGen        


@unittest.skipIf(not HUGGINGFACE_MODELS, reason="Hugging face transformers could not be imported")
class TestClassifierHuggingfaceElectraGen(TestClassifierTextCNN):
    base_model = HFElectraGen


@unittest.skipIf(not HUGGINGFACE_MODELS, reason="Hugging face transformers could not be imported")
class TestComparisonHuggingfaceElectraGen(TestComparisonTextCNN):
    base_model = HFElectraGen

@unittest.skipIf(not HUGGINGFACE_MODELS, reason="Hugging face transformers could not be imported")
class TestSequenceHuggingfaceElectraDiscrim(TestSequenceLabelerTCN):
    base_model = HFElectraDiscrim        


@unittest.skipIf(not HUGGINGFACE_MODELS, reason="Hugging face transformers could not be imported")
class TestClassifierHuggingfaceElectraDiscrim(TestClassifierTextCNN):
    base_model = HFElectraDiscrim


@unittest.skipIf(not HUGGINGFACE_MODELS, reason="Hugging face transformers could not be imported")
class TestComparisonHuggingfaceElectraDiscrim(TestComparisonTextCNN):
    base_model = HFElectraDiscrim


@unittest.skipIf(not HUGGINGFACE_MODELS, reason="Hugging face transformers could not be imported")
class TestSequenceHuggingfaceBERT(TestSequenceLabelerTCN):
    base_model = HFBert

    def test_low_memory_mode(self):
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1
        )
        train_texts = [t * 10 for t in train_texts]
        self.model.config.low_memory_mode = True
        self.model.config.batch_size = 32

        self.model.fit(train_texts * 10, train_annotations * 10)

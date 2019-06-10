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
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

from finetune import Classifier, Regressor, DeploymentModel, SequenceLabeler
from finetune.base_models import GPTModel, 
from finetune.datasets import generic_download
from finetune.config import get_config
from finetune.errors import FinetuneError

SST_FILENAME = "SST-binary.csv"

SKIP_LM_TESTS = get_config().base_model.is_bidirectional

class TestClassifier(unittest.TestCase):
    n_sample = 20
    classifier_path = "tests/saved-models/deployment_classifier.jl"
    comparison_regressor_path = "tests/saved-models/deployment_comparison_regressor.jl"
    sequence_labeler_path = "tests/saved-models/deployment_sequence_labeler.jl"
    classifier_dataset_path = os.path.join(
        'Data', 'Classify', 'SST-binary.csv'
    )
    sequence_dataset_path = os.path.join(
        'Data', 'Sequence', 'reuters.xml'
    )
    processed_path = os.path.join('Data', 'Sequence', 'reuters.json')


    @classmethod
    def _download_data(cls):
        
        #Download Stanford Sentiment Treebank to data directory
        
        path = Path(cls.classifier_dataset_path)
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME
        )

        #Download Reuters Dataset to enso `data` directory
        
        path = Path(cls._path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(cls.dataset_path):
            url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
            r = requests.get(url)
            with open(cls.sequence_dataset_path, "wb") as fp:
                fp.write(r.content)

        with codecs.open(cls.sequence_dataset_path, "r", "utf-8") as infile:
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


        with open(cls.processed_path, 'wt') as fp:
            json.dump((docs, docs_labels), fp)


    @classmethod
    def setUpClass(cls):
        cls._download_data()

    def setUp(self):
        #dataset preparation
        self.classifier_dataset = pd.read_csv(self.classifier_path, nrows=self.n_sample * 3)

        with open(self.processed_path, 'rt') as fp:
            self.texts, self.labels = json.load(fp)

        self.animals = ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken", "guinea pig", "donkey", "turkey", "duck", "camel", "goose", "llama", "rabbit", "fox"]
        self.numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]


        #train and save classifier for later use
        cl = Classifier()
        train_sample = self.classifier_dataset.sample(n=self.n_sample)
        cl.fit(train_sample.Text, train_sample.Target)
        cl.save(self.classifier_path)


        #train and save comparison regressor for use
        cr = ComparisonRegressor()
 
        n_per = 150
        similar = []
        different = []
        for dataset in [self.animals, self.numbers]:
            for i in range(n_per // 2):
                similar.append([random.choice(dataset), random.choice(dataset)])
        for i in range(n_per):
            different.append([random.choice(animals), random.choice(numbers)])

        targets = np.asarray([1] * len(similar) + [0] * len(different))
        data = similar + different

        self.x_tr, self.x_te, self.t_tr, self.t_te = train_test_split(data, targets, test_size=0.3, random_state=42)
        cr.finetune(self.x_tr, self.t_tr)
        cr.save(self.comparison_regressor_path)

        #train and save sequence labeler for later use
        s = SequenceLabeler()
        s.fit(self.texts * 10, self.labels * 10)
        s.save(self.sequence_labeler_path)

        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn("tests/saved-models still exists, it is possible that some test is not cleaning up properly.")
            pass

    def tearDown(self):
        shutil.rmtree("tests/saved-models/")

    def default_config(self, **kwargs):
        defaults = {
            'batch_size': 2,
            'max_length': 128,
            'n_epochs': 1,
        }
        defaults.update(kwargs)
        return dict(get_config(**defaults))

    def test_cached_predict(self):
        """
        Ensure second call to predict is faster than first
        """
        model = DeploymentModel(base_model=GPTModel, **self.default_config())
        model.load_featurizer()
        model.load_trainables(self.classifier_path)

        valid_sample = self.classifier_dataset.sample(n=self.n_sample)
        start = time.time()
        model.predict(valid_sample.Text[:1].values)
        first = time.time()
        model.predict(valid_sample.Text[:1].values)
        second = time.time()

        first_prediction_time = (first - start)
        second_prediction_time = (second - first)
        self.assertLess(second_prediction_time, first_prediction_time / 2.)

    def test_switching_models(self):
        """
        Ensure model can switch out weights without erroring out
        """
        model = DeploymentModel(base_model=GPTModel, **self.default_config())
        model.load_featurizer()
        #test transitioning from any of [sequence labeling, comparison, default] to any other
        model.load_trainables(self.classifier_path)
        model.load_trainables(self.comparison_regressor_path)
        model.load_trainables(self.sequence_labeler_path)
        model.load_trainables(self.classifier_path)
        model.load_trainables(self.sequence_labeler_path)
        model.load_trainables(self.comparison_regressor_path)
        model.load_trainables(self.classifier_path)

    def test_reasonable_predictions(self):
        """
        Ensure model produces reasonable predictions after loading weights
        """
        model = DeploymentModel(base_model=GPTModel, **self.default_config())
        model.load_featurizer()
        model.load_trainables(self.classifier_path)

        n_per_class = (self.n_sample * 5)
        trX = ['cat'] * n_per_class + ['finance'] * n_per_class
        trY = copy(trX)
        teX = ['feline'] + ['investment']
        teY = ['cat'] + ['finance']
        model.fit(trX, trY)
        predY = model.predict(teX)
        self.assertEqual(accuracy_score(teY, predY), 1.00)

        model.load_trainables(self.comparison_regressor_path)
        predictions = model.predict(self.x_te)
        mse = np.mean([(pred - true)**2 for pred, true in zip(predictions, self.t_te)])
        naive_baseline = max(np.mean(targets == 1), np.mean(targets == 0))
        naive_baseline_mse = np.mean([(naive_baseline - true)**2 for true in self.t_te])
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions[0], np.float32)
        self.assertGreater(naive_baseline_mse, mse)

        model.load_trainables(self.sequence_labeler_path)

        predictions = model.predict(test_sequence)
        self.assertTrue(1 <= len(predictions[0]) <= 3)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in predictions[0]))

    def test_fast_switch(self):
        model = DeploymentModel(base_model=GPTModel, **self.default_config())
        model.load_featurizer()
        model.load_trainables(self.classifier_path)
        model.predict('finetune')

        start = time.time()
        model.load_trainables(self.sequence_labeler_path)
        predictions = model.predict(['finetune sequence'])
        end = time.time()
        self.assertGreater(2.5, end - start)

        start = time.time()
        model.load_trainables(self.comparison_regressor_path)
        predictions = model.predict([['finetune', 'compare']])
        end = time.time()
        self.assertGreater(2.5, end - start)



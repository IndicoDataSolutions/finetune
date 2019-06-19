import os
import unittest
import logging
import shutil
import string
import gc
import json
from copy import copy
import random
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
from sklearn.model_selection import train_test_split

from finetune import Classifier, Regressor, DeploymentModel, SequenceLabeler, ComparisonRegressor
from finetune.base_models import GPTModel, BERTModelCased
from finetune.datasets import generic_download
from finetune.config import get_config
from finetune.errors import FinetuneError

SST_FILENAME = "SST-binary.csv"

class TestDeploymentModel(unittest.TestCase):
    n_sample = 20
    do_comparison = True
    bert_intermediate_size = None
    base_model = GPTModel

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
        
        path = Path(cls.sequence_dataset_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(cls.sequence_dataset_path):
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

        try:
            os.mkdir("tests/saved-models-bert")
        except FileExistsError:
            warnings.warn("tests/saved-models still exists, it is possible that some test is not cleaning up properly.")
            pass

        #dataset preparation
        self.classifier_dataset = pd.read_csv(self.classifier_dataset_path, nrows=self.n_sample * 10)

        path = os.path.join(os.path.dirname(__file__), "testdata.json")
        with open(path, 'rt') as fp:
            self.texts, self.labels = json.load(fp)

        self.animals = ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken", "guinea pig", "donkey", "turkey", "duck", "camel", "goose", "llama", "rabbit", "fox"]
        self.numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]
        
        #train and save sequence labeler for later use
        
        try:
            self.s = SequenceLabeler.load(self.sequence_labeler_path, **self.default_seq_config())
        except Exception as e:
            self.s = SequenceLabeler(**self.default_seq_config())
            self.s.fit(self.texts * 10, self.labels * 10)
            self.s.save(self.sequence_labeler_path)
        
        #train and save classifier for later use
        
        train_sample = self.classifier_dataset.sample(n=self.n_sample*10)
        try:
            self.cl = Classifier.load(self.classifier_path)
        except Exception as e:
            self.cl = Classifier(**self.default_config())
            self.cl.fit(train_sample.Text, train_sample.Target)
            self.cl.save(self.classifier_path)

        
        if self.do_comparison:
            #train and save comparison regressor for use
            self.cr = ComparisonRegressor()
    
            n_per = 150
            similar = []
            different = []
            for dataset in [self.animals, self.numbers]:
                for i in range(n_per // 2):
                    similar.append([random.choice(dataset), random.choice(dataset)])
            for i in range(n_per):
                different.append([random.choice(self.animals), random.choice(self.numbers)])

            targets = np.asarray([1] * len(similar) + [0] * len(different))
            data = similar + different

            self.x_tr, self.x_te, self.t_tr, self.t_te = train_test_split(data, targets, test_size=0.3, random_state=42)
            
            try:
                self.cr = ComparisonRegressor.load(self.comparison_regressor_path, **self.default_config())
            except Exception as e:
                self.cr = ComparisonRegressor(**self.default_config())
                self.cr.fit(self.x_tr, self.t_tr)
                self.cr.save(self.comparison_regressor_path)
            
    def tearDown(self):
        shutil.rmtree("tests/saved-models/")

    def default_config(self, **kwargs):
        defaults = {
            'batch_size': 2,
            'max_length': 128,
            'n_epochs': 3,
            'adapter_size': 64,
            'base_model': self.base_model
        }
        defaults.update(kwargs)
        return dict(get_config(**defaults))

    def default_seq_config(self, **kwargs):
        d = dict(
            batch_size=2,
            max_length=256,
            lm_loss_coef=0.0,
            val_size=0,
            adapter_size=64,
            base_model=self.base_model,
            interpolate_pos_embed=False,
        )
        d.update(**kwargs)
        return d

    def dtest_cached_predict(self):
        """
        Ensure second call to predict is faster than first, including loading.
        """
        model = DeploymentModel(featurizer=self.base_model, **self.default_config())
        model.load_featurizer()

        valid_sample = self.classifier_dataset.sample(n=self.n_sample)
        start = time.time()
        model.load_trainables(self.classifier_path)
        model.predict(valid_sample.Text[:1].values)
        first = time.time()
        model.predict(valid_sample.Text[:1].values)
        second = time.time()

        first_prediction_time = (first - start)
        second_prediction_time = (second - first)
        self.assertLess(second_prediction_time, first_prediction_time)

    def dtest_switching_models(self):
        """
        Ensure model can switch out weights without erroring out
        """
        model = DeploymentModel(featurizer=self.base_model, **self.default_config())
        model.load_featurizer()
        #test transitioning from any of [sequence labeling, comparison, default] to any other
        model.load_trainables(self.classifier_path)
        if self.do_comparison:
            model.load_trainables(self.comparison_regressor_path)
        model.load_trainables(self.sequence_labeler_path)
        model.load_trainables(self.classifier_path)
        model.load_trainables(self.sequence_labeler_path)
        if self.do_comparison:
            model.load_trainables(self.comparison_regressor_path)
        model.load_trainables(self.classifier_path)

    def test_reasonable_predictions(self):
        """
        Ensure model produces reasonable predictions after loading weights
        """
        
        model = DeploymentModel(featurizer=self.base_model, **self.default_seq_config())
        model.load_featurizer()
        
        #test same output as weights loaded with Classifier model
        valid_sample = self.classifier_dataset.sample(n=self.n_sample)
        model.load_trainables(self.classifier_path)
        deployment_preds = model.predict_proba(valid_sample.Text.values)
        classifier_preds = self.cl.predict_proba(valid_sample.Text.values)
        
        for c_pred, d_pred in zip(classifier_preds, deployment_preds):
            self.assertTrue(list(c_pred.keys()) == list(d_pred.keys()))
            for c_pred_val, d_pred_val in zip(c_pred.values(), d_pred.values()):
                np.testing.assert_almost_equal(c_pred_val, d_pred_val, decimal=2)
        
        if self.do_comparison:
            #test same output as weights loaded with Comparison Regressor model
            model.load_trainables(self.comparison_regressor_path)
            compregressor = ComparisonRegressor.load(self.comparison_regressor_path,  **self.default_seq_config())
            deployment_preds = model.predict(self.x_te)
            compregressor_preds = compregressor.predict(self.x_te)

            for c_pred, d_pred in zip(compregressor_preds, deployment_preds):
                np.testing.assert_almost_equal(c_pred, d_pred, decimal=2)

        '''
        #test reasonable output for weights loaded with Sequence Labeler model
        test_sequence = ["I am a dog. A dog that's incredibly bright. I can talk, read, and write! "]
        #model.load_trainables(self.sequence_labeler_path)
        #deployment_preds = model.predict_proba(test_sequence)
        sequence_preds = self.s.predict_proba(test_sequence)
        differing = [s for s in model.config if model.config[s] != self.s.config[s]]
        for key in differing:
            print(key)
            print(model.config[key])
            print(self.s.config[key])
        print('')
        #print(deployment_preds)
        print(sequence_preds)
        self.assertTrue(1 <= len(deployment_preds[0]) <= 3)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in deployment_preds[0]))
        '''

    def dtest_large_predict(self):
        """
        Ensure model does not have OOM issues with large inputs for inference
        """
        large_dataset = self.animals*100
        model = DeploymentModel(featurizer=self.base_model, **self.default_config())
        model.load_featurizer()
        model.load_trainables(self.classifier_path)
        model.predict(large_dataset)


    def dtest_fast_switch(self):
        """
        Ensure model can load/reload weights and predict in reasonable time
        """
        model = DeploymentModel(featurizer=self.base_model, **self.default_config())
        model.load_featurizer()
        model.load_trainables(self.classifier_path)
        model.predict(['finetune'])

        start = time.time()
        model.load_trainables(self.sequence_labeler_path)
        predictions = model.predict(['finetune sequence'])
        end = time.time()
        self.assertGreater(2.5, end - start)

        if self.do_comparison:
            start = time.time()
            model.load_trainables(self.comparison_regressor_path)
            predictions = model.predict([['finetune', 'compare']])
            end = time.time()
            self.assertGreater(2.5, end - start)
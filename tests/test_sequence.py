import os
import unittest
import logging
from copy import copy
from pathlib import Path
import codecs
import json
import random
import time

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from finetune import SequenceLabeler
from finetune.config import get_config
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.util.metrics import (
    sequence_labeling_token_precision, sequence_labeling_token_recall,
    sequence_labeling_overlap_precision, sequence_labeling_overlap_recall
)

SKIP_LM_TESTS = get_config().base_model.is_bidirectional


class TestSequenceLabeler(unittest.TestCase):

    n_sample = 100
    dataset_path = os.path.join(
        'Data', 'Sequence', 'reuters.xml'
    )
    processed_path = os.path.join('Data', 'Sequence', 'reuters.json')

    @classmethod
    def _download_reuters(cls):
        """
        Download Stanford Sentiment Treebank to enso `data` directory
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


        with open(cls.processed_path, 'wt') as fp:
            json.dump((docs, docs_labels), fp)


    @classmethod
    def setUpClass(cls):
        cls._download_reuters()

    def default_config(self, **kwargs):
        d = dict(
            batch_size=2,
            max_length=256,
            lm_loss_coef=0.0,
            val_size=0,
            interpolate_pos_embed=False,
        )
        d.update(**kwargs)
        return d

    def setUp(self):
        self.save_file = 'tests/saved-models/test-save-load'
        random.seed(42)
        np.random.seed(42)
        with open(self.processed_path, 'rt') as fp:
            self.texts, self.labels = json.load(fp)

        self.model = SequenceLabeler(
            **self.default_config()
        )

    @pytest.mark.skipif(SKIP_LM_TESTS, reason="Bidirectional models do not yet support LM functions")
    def test_fit_lm_only(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, self.texts, self.labels,
                                                         none_value=self.model.config.pad_token)
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(texts, annotations, test_size=0.1)
        self.model.fit(train_texts)
        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)
        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]['confidence'], dict)
        token_precision = sequence_labeling_token_precision(test_annotations, predictions)
        token_recall = sequence_labeling_token_recall(test_annotations, predictions)
        overlap_precision = sequence_labeling_overlap_precision(test_annotations, predictions)
        overlap_recall = sequence_labeling_overlap_recall(test_annotations, predictions)
        self.assertIn('Named Entity', token_precision)
        self.assertIn('Named Entity', token_recall)
        self.assertIn('Named Entity', overlap_precision)
        self.assertIn('Named Entity', overlap_recall)
        self.model.save(self.save_file)
        model = SequenceLabeler.load(self.save_file)
        predictions = model.predict(test_texts)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        Ensure class reweighting behaves as intended
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, self.texts, self.labels,
                                                         none_value=self.model.config.pad_token)
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1, random_state=42
        )

        reweighted_model = SequenceLabeler(
            **self.default_config(class_weights={'Named Entity': 10.})
        )
        reweighted_model.fit(train_texts, train_annotations)
        reweighted_predictions = reweighted_model.predict(test_texts)
        reweighted_token_recall = sequence_labeling_token_recall(test_annotations, reweighted_predictions)

        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)

        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]['confidence'], dict)

        token_precision = sequence_labeling_token_precision(test_annotations, predictions)
        token_recall = sequence_labeling_token_recall(test_annotations, predictions)
        overlap_precision = sequence_labeling_overlap_precision(test_annotations, predictions)
        overlap_recall = sequence_labeling_overlap_recall(test_annotations, predictions)

        self.assertIn('Named Entity', token_precision)
        self.assertIn('Named Entity', token_recall)
        self.assertIn('Named Entity', overlap_precision)
        self.assertIn('Named Entity', overlap_recall)

        self.model.save(self.save_file)

        self.assertGreater(reweighted_token_recall['Named Entity'], token_recall['Named Entity'])

    def test_cached_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, self.texts, self.labels,
                                                         none_value=self.model.config.pad_token)
        train_texts, test_texts, train_annotations, _ = train_test_split(texts, annotations, test_size=0.1)
        self.model.fit(train_texts, train_annotations)
        
        self.model.config.chunk_long_sequences = True
        self.model.config.max_length = 128

        uncached_preds = self.model.predict(test_texts[:1])

        with self.model.cached_predict():
            start = time.time()
            self.model.predict(test_texts[:1])
            first = time.time()
            self.model.predict(test_texts[:1])
            second = time.time()
            preds = self.model.predict(test_texts[:1])
            assert len(preds) == 1
            preds = self.model.predict(test_texts[:2])
            assert len(preds) == 2

        for uncached_pred, cached_pred in zip(uncached_preds, preds):
            self.assertEqual(str(uncached_pred), str(cached_pred))

        first_prediction_time = (first - start)
        second_prediction_time = (second - first)
        self.assertLess(second_prediction_time, first_prediction_time / 2.)

    def test_reasonable_predictions(self):
        test_sequence = ["I am a dog. A dog that's incredibly bright. I can talk, read, and write!"]
        path = os.path.join(os.path.dirname(__file__), "data", "testdata.json")

        # test ValueError raised when raw text is passed along with character idxs and doesn't match
        with self.assertRaises(ValueError):
            self.model.fit(["Text about a dog."], [[{"start": 0, "end": 5, "text": "cat", "label": "dog"}]])

        with open(path, "rt") as fp:
            text, labels = json.load(fp)

        self.model.fit(text * 10, labels * 10)

        predictions = self.model.predict(test_sequence)
        self.assertTrue(1 <= len(predictions[0]) <= 3)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in predictions[0]))

        predictions = self.model.predict(test_sequence)
        self.assertTrue(1 <= len(predictions[0]) <= 3)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in predictions[0]))

    def test_chunk_long_sequences(self):
        test_sequence = ["I am a dog. A dog that's incredibly bright. I can talk, read, and write! " * 10]
        path = os.path.join(os.path.dirname(__file__), "data", "testdata.json")

        # test ValueError raised when raw text is passed along with character idxs and doesn't match
        self.model.config.chunk_long_sequences = True
        self.model.config.max_length = 18
        with self.assertRaises(ValueError):
            self.model.fit(["Text about a dog."], [[{"start": 0, "end": 5, "text": "cat", "label": "dog"}]])

        with open(path, "rt") as fp:
            text, labels = json.load(fp)

        self.model.finetune(text * 10, labels * 10)

        predictions = self.model.predict(test_sequence)
        self.assertEqual(len(predictions[0]), 20)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in predictions[0]))

    def test_fit_predict_multi_model(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        self.model = SequenceLabeler(batch_size=2, max_length=256, lm_loss_coef=0.0, multi_label_sequences=True)
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, self.texts, self.labels,
                                                         none_value=self.model.config.pad_token)
        train_texts, test_texts, train_annotations, _ = train_test_split(texts, annotations, test_size=0.1)
        self.model.fit(train_texts, train_annotations)
        self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)
        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]['confidence'], dict)
        self.model.save(self.save_file)
        model = SequenceLabeler.load(self.save_file)
        model.predict(test_texts)

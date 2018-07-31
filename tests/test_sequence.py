import os
import unittest
import logging
from copy import copy
from pathlib import Path
import codecs
import json

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from finetune import SequenceLabeler
from finetune.utils import indico_to_finetune_sequence, finetune_to_indico_sequence


class TestSequenceLabeler(unittest.TestCase):

    n_sample = 100
    n_hidden = 768
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
            soup = bs(infile, "html5lib")

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

    def setUp(self):
        self.save_file = 'tests/saved-models/test-save-load'

        with open(self.processed_path, 'rt') as fp:
            self.texts, self.labels = json.load(fp)
        
        tf.reset_default_graph()

        self.model = SequenceLabeler(batch_size=2, max_length=256, lm_loss_coef=0.0, verbose=False)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, self.texts, self.labels)
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(texts, annotations, test_size=0.1)
        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)
        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], tuple)
        self.assertIsInstance(probas[0][0][1], dict)
        self.model.save(self.save_file)
        model = SequenceLabeler.load(self.save_file)
        predictions = model.predict(test_texts)

    def test_reasonable_predictions(self):
        test_sequence = ["I am a dog. A dog that's incredibly bright. I can talk, read, and write!"]
        path = os.path.join(os.path.dirname(__file__), "testdata.json")

        # test ValueError raised when raw text is passed along with character idxs and doesn't match
        with self.assertRaises(ValueError):
            self.model.fit(["Text about a dog.", {"start": 0, "end": 5, "text": "cat", "label": "dog"}])

        with open(path, "rt") as fp:
            text, labels = json.load(fp)
        self.model.finetune(text * 10, labels * 10)
        
        predictions = self.model.predict(test_sequence)
        self.assertTrue(1 <= len(predictions[0]) <= 3)
        self.assertTrue(any(pred["text"] == "dog" for pred in predictions[0]))

        self.model.config.subtoken_predictions = True
        predictions = self.model.predict(test_sequence)
        self.assertTrue(1 <= len(predictions[0]) <= 3)
        self.assertTrue(any(pred["text"] == "dog" for pred in predictions[0]))

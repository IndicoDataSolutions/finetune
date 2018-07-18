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
import enso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from finetune import SequenceLabeler
from finetune.config import get_hparams
from finetune.utils import indico_to_finetune_sequence, finetune_to_indico_sequence


class TestSequenceLabeler(unittest.TestCase):

    n_sample = 100
    n_hidden = 768
    dataset_path = os.path.join(
        enso.config.DATA_DIRECTORY, 'Sequence', 'reuters.xml'
    )
    processed_path = os.path.join(enso.config.DATA_DIRECTORY, 'Sequence', 'reuters.json')

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
        
        # if not os.path.exists(cls.processed_path):

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
        self.save_file_autosave = 'tests/saved-models/autosave_path'
        self.save_file = 'tests/saved-models/test-save-load'

        with open(self.processed_path, 'rt') as fp:
            self.texts, self.labels = json.load(fp)
        
        tf.reset_default_graph()

        hparams = get_hparams(batch_size=2, max_length=256)
        self.model = SequenceLabeler(hparams=hparams, verbose=False, autosave_path=self.save_file_autosave)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        texts, annotations = finetune_to_indico_sequence(self.texts, self.labels)
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(texts, annotations)
        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        self.model.save(self.save_file_autosave)

import os
import unittest
import logging
from copy import copy
from pathlib import Path

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import enso
from sklearn.metrics import accuracy_score
import requests
from finetune import LanguageModelGeneralAPI

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import json

SST_FILENAME = "SST-binary.csv"


class TestLanguageModelSequenceLabel(unittest.TestCase):

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
        if not os.path.exists(cls.dataset_path):
            url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
            r = requests.get(url)
            with open(cls.dataset_path, "wb") as fp:
                fp.write(r.content)

            with codecs.open(cls.dataset_path, "r", "utf-8") as infile:
                soup = bs(infile, "html5lib")

            docs = []
            for elem in soup.find_all("document"):
                texts = []

                # Loop through each child of the element under "textwithnamedentities"
                for c in elem.find("textwithnamedentities").children:
                    if type(c) == Tag:
                        if c.name == "namedentityintext":
                            label = "N"  # part of a named entity
                        else:
                            label = "I"  # irrelevant word
                        texts.append([c.text, label])
                docs.append(texts)
            with open(cls.processed_path, 'wt') as fp:
                json.dump(docs, fp)


    @classmethod
    def setUpClass(cls):
        cls._download_reuters()

    def setUp(self):
        with open(self.processed_path, 'rt') as fp:
            self.dataset = json.load(fp)
        tf.reset_default_graph()

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """
        save_file_autosave = 'tests/saved-models/autosave_path'

        model = LanguageModelGeneralAPI(verbose=False, autosave_path=save_file_autosave)
        model.fit(list(zip(self.dataset)), None)
        predictions = model.predict(list(zip(self.dataset)))


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
from pytest import approx

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from finetune.target_models.self_suprevised import SSLLabeler
from finetune.base_models import GPT
from finetune.config import get_config
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.util.metrics import (
    sequence_labeling_token_precision, sequence_labeling_token_recall,
    sequence_labeling_overlap_precision, sequence_labeling_overlap_recall
)
from finetune.input_pipeline import InputMode

SKIP_LM_TESTS = get_config().base_model.is_bidirectional


class TestSSLLabeler(unittest.TestCase):

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
            base_model=GPT,
            batch_size=2,
            max_length=256,
            lm_loss_coef=0.0,
            val_size=0,
        )
        d.update(**kwargs)
        return d

    def setUp(self):
        self.save_file = 'tests/saved-models/test-save-load'
        random.seed(42)
        np.random.seed(42)
        with open(self.processed_path, 'rt') as fp:
            self.texts, self.labels = json.load(fp)

        self.model = SSLLabeler(
            **self.default_config()
        )

    def test_pipeline(self):
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, self.texts, self.labels,
                                                         none_value=self.model.config.pad_token)
        cut = len(texts) // 5
        Xs = texts[:cut]
        Ys = annotations[:cut]
        Us = texts[cut:]
        print(f"Length of X: {len(Xs)}")
        print(f"Length of U: {len(Us)}")
        x_list = self.model.input_pipeline.zip_list_to_dict(X=Xs, Y=Ys)
        u_list = self.model.input_pipeline.zip_list_to_dict(X=Us)
        dataset = self.model.input_pipeline.get_dataset_from_list(x_list,
                                                                  input_mode=InputMode.TRAIN,
                                                                  u_data_list=u_list)
        dataset = dataset["train_dataset"]()
        print(list(dataset.as_numpy_iterator())[:3])

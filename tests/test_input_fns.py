import os
import unittest
import logging
import shutil
import string
import gc
from copy import copy
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
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
from finetune.util.input_utils import InputMode

SST_FILENAME = "SST-binary.csv"

def dummy_make_dataset_fn(**kwargs):
    return kwargs

class TestInputFns(unittest.TestCase):
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
        self.dataset = pd.read_csv(self.dataset_path)
        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn(
                "tests/saved-models still exists, it is possible that some test is not cleaning up properly."
            )
            pass

    def tearDown(self):
        shutil.rmtree("tests/saved-models/")

    @patch('finetune.input_pipeline.batch_dataset')
    def test_input_fns(self, mock_batch_dataset):
        for setting in [
                {
                    "val_size": 0.3,
                    "chunk": True,
                    "train_len": 407,
                    "val_len": 147
                },
                {
                    "val_size": 0.3,
                    "chunk": False,
                    "train_len": 70,
                    "val_len": 30
		},
                {
                    "val_size":	0.0,
                    "chunk": True,
                    "train_len": 554,
                    "val_len": 0
		},
                {
                    "val_size":	0.0,
                    "chunk": False,
                    "train_len": 100,
                    "val_len": 0
                },
        ]:
            model = Classifier(max_length=10, chunk_long_sequences=setting["chunk"], batch_size=3, n_epochs=7, val_size=setting["val_size"])
            model.input_pipeline.make_dataset_fn = dummy_make_dataset_fn
            train_sample = self.dataset.sample(n=100)
            zipped_data = model.input_pipeline.zip_list_to_dict(X=train_sample.Text.values, Y=train_sample.Target.values)
            dummy_hook = "DUMMY HOOK"
            output = model.input_pipeline.get_dataset_from_list(zipped_data, InputMode.TRAIN, update_hook=dummy_hook)
            calls = mock_batch_dataset.call_args_list
            (train_ds,), train_call = calls[-2]
            (val_ds,), val_call = calls[-1]
            self.assertEqual(train_call["batch_size"], 3)
            self.assertEqual(val_call["batch_size"], 3)
            self.assertEqual(train_call["n_epochs"], 7)
            self.assertNotIn("n_epochs", val_call)
            
            self.assertEqual(train_ds["types"], val_ds["types"])
            self.assertEqual(train_ds["shapes"], val_ds["shapes"])
            
            self.assertEqual(train_ds["tqdm_mode"], "train")
            self.assertEqual(val_ds["tqdm_mode"], "evaluate")
            
            self.assertEqual(train_ds['update_hook'], dummy_hook)
            self.assertNotIn("update_hook", val_ds)
            
            train_data = list(train_ds["data_fn"]())
            val_data = list(val_ds["data_fn"]())
            
            self.assertEqual(len(train_data), setting["train_len"])
            self.assertEqual(len(val_data), setting["val_len"])

            train_tokens = [tr[0]["tokens"].tolist() for tr in train_data]
            test_tokens = [te[0]["tokens"].tolist() for te in val_data]
            for tr in train_tokens:
                self.assertTrue(tr not in test_tokens)

            self.assertEqual(len(train_data), model.config.dataset_size)

            

        

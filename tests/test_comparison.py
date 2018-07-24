import os
import unittest
import logging
import shutil
from copy import copy
from pathlib import Path
from unittest.mock import MagicMock
import warnings

# prevent excessive warning logs 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from finetune import Comparison
from finetune.datasets import generic_download
from finetune.config import get_config

SST_FILENAME = "SST-binary.csv"


class TestComparison(unittest.TestCase):
    n_sample = 20
    n_hidden = 768
    dataset_path = os.path.join(
        'Data', 'Classify', 'SST-binary.csv'
    )

    def default_config(self, **kwargs):
        return get_config(
            batch_size=2,
            max_length=128,
            n_epochs=1,
            verbose=False,
            **kwargs
        )

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Comparison(config=self.default_config())
        model.fit(["Indico is the best"]*10, ["Indico is the bestestestest"]*10, np.array([0]*10))

        predictions = model.predict(["Is indico the best?"], ["Indico is the bestestestest"])
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))
import os
import unittest
import shutil
import json
import random
import time
from pathlib import Path
import warnings

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from finetune.base_models import BERTModelCased, GPT2Model, RoBERTa
from finetune import Classifier, DeploymentModel, SequenceLabeler, ComparisonRegressor
from finetune.base_models import GPTModel
from finetune.datasets import generic_download
from finetune.scheduler import Scheduler

class TestDeploymentModel(unittest.TestCase):

    folder = "tests/saved-models"
    model1 = "1.jl"
    model2 = "2.jl"

    @classmethod
    def setUpClass(cls):
        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn(
                "tests/saved-models still exists, it is possible that some test is not cleaning up properly."
            )
            pass
        model = Classifier(base_model=RoBERTa)
        model.fit(["A", "B"], ["a", "b"])
        model.save(os.path.join(cls.folder, cls.model1))
        model.save(os.path.join(cls.folder, cls.model2))

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("tests/saved-models/")
        
    def test_scheduler(self):
        m1 = os.path.join(self.folder, self.model1)
        m2 = os.path.join(self.folder, self.model2)
        shed = Scheduler()

        # Checks model caching
        time_pre = time.time()
        pred1a = shed.predict(m1, ["A"])
        time_mid = time.time()
        pred1b = shed.predict(m1, ["A"])
        time_end = time.time()
        self.assertLess(time_end - time_mid, time_mid - time_pre - 1) # Assert that it is at least 1 second quicker
        self.assertEqual(pred1a, pred1b)
        shed.predict(m2, ["A"]) # Load another model.
        time2_start = time.time()
        pred1b = shed.predict(m1, ["A"])
        time2_end = time.time()
        self.assertLess(time2_end - time2_start, time_mid - time_pre - 1) # Assert that it is still quicker.
        

        

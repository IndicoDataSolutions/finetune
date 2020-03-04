import unittest
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from finetune.base_models import OSCAR
from finetune import Classifier
from finetune.util.metrics import read_eval_metrics


class TestOscarFeatures(unittest.TestCase):

    def test_in_memory_finetune(self):
        SST_FILENAME = "SST-binary.csv"
        DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
        dataframe = pd.read_csv(DATA_PATH, nrows=100).dropna()
        trainX, testX, trainY, testY = train_test_split(list(dataframe.Text.values), list(dataframe.Target.values),
                                                        test_size=0.3, random_state=42)
        in_memory_finetune = [
            {
                "config": {"n_epochs": 1, "max_length": 64},
                "X": trainX,
                "Y": trainY,
                "X_test": testX,
                "Y_test": testY,
                "name": "sst-b",
                "every_n_iter": 30
            }
        ]
        model = Classifier(in_memory_finetune=in_memory_finetune, max_length=64)
        model.fit(trainX, trainY)
        metrics = read_eval_metrics(os.path.join(model.estimator_dir, "finetuning"))
        for step, metric in metrics.items():
            self.assertEqual(len(metric), 2) # train and test
            for key, value in metric.items():
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
                self.assertIn("finetuning", key)

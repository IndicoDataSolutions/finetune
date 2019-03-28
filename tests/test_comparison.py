import os
import unittest
import warnings
import random

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from finetune import Comparison
from finetune.util import list_transpose
import random

SST_FILENAME = "SST-binary.csv"


class TestComparison(unittest.TestCase):
    n_sample = 20
    dataset_path = os.path.join(
        'Data', 'Classify', 'SST-binary.csv'
    )

    def default_config(self, **kwargs):
        d = dict(
            batch_size=2,
            max_length=128,
            n_epochs=1,
            l2_reg=0,
            lm_loss_coef=0.,
            val_size=0.,
        )
        d.update(kwargs)
        return d

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Comparison(**self.default_config())
        n_samples = 10
        model.fit(
            [["Transformers was a terrible movie but a great model", "Transformers are a great model but a terrible movie"]] * n_samples,
            ['yes'] * n_samples
        )

        test_data = [["Transformers was a terrible movie but a great model", "Transformers are a great model but a terrible movie"]]

        predictions = model.predict(test_data)
        for prediction in predictions:
            self.assertIsInstance(prediction, (str, bytes))

        probabilities = model.predict_proba(test_data)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_reasonable_predictions(self):
        model = Comparison(**self.default_config())

        # fake dataset generation
        animals = ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken", "guinea pig", "donkey", "turkey", "duck", "camel", "goose", "llama", "rabbit", "fox"]
        numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]

        n_per = 100
        similar = []
        different = []
        for dataset in [animals, numbers]:
            for i in range(n_per // 2):
                similar.append([random.choice(dataset), random.choice(dataset)])
        for i in range(n_per):
            different.append([random.choice(animals), random.choice(numbers)])

        targets = np.asarray(["similar"] * len(similar) + ["different"] * len(different))
        data = similar + different

        x_tr, x_te, t_tr, t_te = train_test_split(data, targets, test_size=0.3)
        model.finetune(x_tr, t_tr)

        predictions = model.predict(x_te)
        accuracy = np.mean([pred == true for pred, true in zip(predictions, t_te)])
        naive_baseline = max(np.mean(targets == "similar"), np.mean(targets == "different"))
        self.assertGreater(accuracy, naive_baseline)

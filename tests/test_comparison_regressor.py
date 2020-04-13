import os
import unittest
import warnings

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random

from sklearn.model_selection import train_test_split
from finetune import ComparisonRegressor


class TestComparisonRegression(unittest.TestCase):
    n_sample = 100

    def default_config(self, **kwargs):
        d = dict(
            batch_size=2,
            max_length=16,
            n_epochs=3,
            val_size=0.,
            l2_reg=0.,
            low_memory_mode=True,
        )
        d.update(kwargs)
        return d

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_reasonable_predictions(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        Test model loss at least outperforms naive baseline
        """
        model = ComparisonRegressor(**self.default_config())

        # fake dataset generation
        animals = ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken", "guinea pig", "donkey", "turkey", "duck", "camel", "goose", "llama", "rabbit", "fox"]
        numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]

        n_per = 150
        similar = []
        different = []
        for dataset in [animals, numbers]:
            for i in range(n_per // 2):
                similar.append([random.choice(dataset), random.choice(dataset)])
        for i in range(n_per):
            different.append([random.choice(animals), random.choice(numbers)])

        targets = np.asarray([1] * len(similar) + [0] * len(different))
        data = similar + different

        x_tr, x_te, t_tr, t_te = train_test_split(data, targets, test_size=0.3, random_state=42)
        model.finetune(x_tr, t_tr)

        predictions = model.predict(x_te)
        mse = np.mean([(pred - true)**2 for pred, true in zip(predictions, t_te)])
        naive_baseline = max(np.mean(targets == 1), np.mean(targets == 0))
        naive_baseline_mse = np.mean([(naive_baseline - true)**2 for true in t_te])
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions[0], np.floating)
        # whether it is float32 or float64 depends on whether it is run on cpu or gpu.

        self.assertGreater(naive_baseline_mse, mse)


if __name__ == '__main__':
    unittest.main()

import os
import unittest
import warnings

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np

from sklearn.model_selection import train_test_split
from finetune.ordinal_regressor import ComparisonOrdinalRegressor, OrdinalRegressor

ANIMALS = ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken", "guinea pig", "donkey", "turkey", "duck", "camel", "goose", "llama", "rabbit", "fox"]
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]


class TestOrdinal(unittest.TestCase):
    n_sample = 100
    n_hidden = 768

    def default_config(self, **kwargs):
        d = dict(
            batch_size=2,
            max_length=128,
            n_epochs=1,
            l2_reg=0,
            lm_coef=0.,
            val_size=0.,
            verbose=False,
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
        model = OrdinalRegressor(n_epochs=2)

        # fake dataset generation
        targets = np.asarray([1] * len(ANIMALS) + [0] * len(NUMBERS))
        data = ANIMALS + NUMBERS

        x_tr, x_te, t_tr, t_te = train_test_split(data, targets, test_size=0.3)
        model.finetune(x_tr, t_tr)

        predictions = model.predict(x_te)
        mse = np.mean([(pred - true)**2 for pred, true in zip(predictions, t_te)])
        naive_baseline = max(np.mean(targets == 1), np.mean(targets == 0))
        naive_baseline_mse = np.mean([(naive_baseline - true)**2 for true in t_te])
        self.assertIsInstance(predictions, list)
        self.assertGreater(naive_baseline_mse, mse)

    def test_reasonable_predictions_comparison(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        Test model loss at least outperforms naive baseline
        """
        model = ComparisonOrdinalRegressor(n_epochs=2, low_memory_mode=True)

        n_per = 50
        similar = []
        different = []
        for dataset in [ANIMALS, NUMBERS]:
            for i in range(n_per // 2):
                similar.append([random.choice(dataset), random.choice(dataset)])
        for i in range(n_per):
            different.append([random.choice(ANIMALS), random.choice(NUMBERS)])

        targets = np.asarray([1] * len(similar) + [0] * len(different))
        data = similar + different

        x_tr, x_te, t_tr, t_te = train_test_split(data, targets, test_size=0.3)
        model.finetune(x_tr, t_tr)

        predictions = model.predict(x_te)
        mse = np.mean([(pred - true)**2 for pred, true in zip(predictions, t_te)])
        naive_baseline = max(np.mean(targets == 1), np.mean(targets == 0))
        naive_baseline_mse = np.mean([(naive_baseline - true)**2 for true in t_te])
        self.assertIsInstance(predictions, list)
        self.assertGreater(naive_baseline_mse, mse)

    def test_reasonable_predictions_unshared_weights(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        Does not analyze model loss since unshared weights perform poorly at
        these low data volumes
        """
        model = OrdinalRegressor(n_epochs=2, shared=False)

        # fake dataset generation

        targets = np.asarray([1] * len(ANIMALS) + [0] * len(NUMBERS))
        data = ANIMALS + NUMBERS

        x_tr, x_te, t_tr, t_te = train_test_split(data, targets, test_size=0.3)
        model.finetune(x_tr, t_tr)

        predictions = model.predict(x_te)
        self.assertIsInstance(predictions, list)

if __name__ == '__main__':
    unittest.main()
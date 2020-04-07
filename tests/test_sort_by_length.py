import unittest
import time
import random

import numpy as np

from finetune import Classifier


class TestSortByLength(unittest.TestCase):

    def test_sort_by_length(self):
        # Construct a fake dataset where lengths vary wildly
        fake_data = ["A"] * 1000 + ["B " * 120] * 48

        # Blend in the long examples
        random.shuffle(fake_data)

        model = Classifier(optimize_for='predict_speed', sort_by_length=False)
        model._cached_predict = True

        # Prime the pipes
        model.featurize(['Test'])

        start = time.time()
        features = model.featurize(fake_data)
        end = time.time()
        total_no_sort = end - start

        model.config.sort_by_length = True
        start = time.time()
        features_sorted = model.featurize(fake_data)
        end = time.time()
        total_sorted = end - start

        assert total_sorted < total_no_sort
        assert np.allclose(features, features_sorted, atol=1e-3)
        

    def test_chunk_long_sequences(self):
        fake_data = ["A"] * 2 + ["B " * 1200] * 2

        random.shuffle(fake_data)

        model = Classifier(optimize_for='predict_speed', chunk_long_sequences=True, sort_by_length=False)

        features = model.featurize(fake_data)
        model.config.sort_by_length = True
        features_sorted = model.featurize(fake_data)
        assert np.allclose(features, features_sorted, atol=1e-3)

        

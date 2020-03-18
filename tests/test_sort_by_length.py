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

        model = Classifier(optimize_for='predict_speed')
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
        

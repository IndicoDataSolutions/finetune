import unittest
import time
import random

import numpy as np

from finetune import Classifier


class TestSortByLength(unittest.TestCase):

    def test_sort_by_length(self):
        # Construct a fake dataset where lengths vary wildly
        fake_data = (["A"] * 15 + ["B " * 120]) * 64

        model = Classifier(optimize_for='predict_speed', sort_by_length=False, predict_batch_size=16)
        model.fit(["A", "A"], ["B", "B"])
        model._cached_predict = True

        # Prime the pipes
        model.featurize(['Test'])

        start = time.time()
        probas = model.predict_proba(fake_data)
        end = time.time()
        total_no_sort = end - start

        model.config.sort_by_length = True
        start = time.time()
        probas_sorted = model.predict_proba(fake_data)
        end = time.time()
        total_sorted = end - start

        print(total_sorted, total_no_sort)

        # This is somewhat arbitrary to require that the time to predict with sorting
        # is twice as fast as without sorting. This is to ensure that sorting actually
        # happens, because otherwise total_sorted can still be less than total_no_sort
        # by a very minimal fraction due to chance. In practice with this curated dataset,
        # the ratio between total_no_sort and total_sorted is >3
        assert total_no_sort / total_sorted > 2
        for pair in zip(probas, probas_sorted):
            assert np.allclose(list(pair[0].values()), list(pair[1].values()), atol=1e-6)

    def test_chunk_long_sequences(self):
        fake_data = ["A"] * 2 + ["B " * 1200] * 2

        random.shuffle(fake_data)

        model = Classifier(optimize_for='predict_speed', chunk_long_sequences=True, sort_by_length=False)
        model.fit(["A", "A"], ["B", "B"])

        probas = model.predict_proba(fake_data)
        model.config.sort_by_length = True
        probas_sorted = model.predict_proba(fake_data)

        for pair in zip(probas, probas_sorted):
            assert np.allclose(list(pair[0].values()), list(pair[1].values()), atol=1e-6)

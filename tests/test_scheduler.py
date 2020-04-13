import os
import unittest
import time
import shutil
import warnings

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from finetune.base_models import RoBERTa, GPT
from finetune import Classifier
from finetune.scheduler import Scheduler


class TestScheduler(unittest.TestCase):

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
        
        model = Classifier(base_model=GPT)
        model.fit(["A", "B"], ["a", "b"])
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
        pred2a = shed.predict(m2, ["A"]) # Load another model.
        self.assertEqual(len(shed.loaded_models), 2)
        time2_start = time.time()
        pred1b = shed.predict(m1, ["A"])
        time2_end = time.time()
        pred2b = shed.predict(m2, ["A"])
        self.assertEqual(pred2a, pred2b)
        self.assertLess(time2_end - time2_start, time_mid - time_pre - 1) # Assert that it is still quicker.
        shed.predict_proba(m1, ["A"])
        shed.featurize(m1, ["A"])
        shed.featurize(m1, ["A"])
        shed.featurize_sequence(m1, ["A"])

    def test_scheduler_max_models(self):
        m1 = os.path.join(self.folder, self.model1)
        m2 = os.path.join(self.folder, self.model2)
        shed = Scheduler(max_models=1)
        time_pre = time.time()
        pred1a = shed.predict(m1, ["A"])
        time_mid = time.time()
        pred1b = shed.predict(m1, ["A"])
        time_end = time.time()
        self.assertLess(time_end - time_mid, time_mid - time_pre - 1) # Assert that it is at least 1 second quicker
        self.assertEqual(pred1a, pred1b)
        pred2a = shed.predict(m2, ["A"]) # Load another model.
        self.assertEqual(len(shed.loaded_models), 1)

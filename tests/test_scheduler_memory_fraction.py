import os
import unittest
import time
import shutil
import warnings

import pytest

from multiprocessing import Process

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from finetune.base_models import RoBERTa, GPT
from finetune import Classifier
from finetune.scheduler import Scheduler


class TestScheduler(unittest.TestCase):

    folder = "tests/saved-models"
    model1 = "1.jl"
    model2 = "2.jl"

    @classmethod
    def _setup(cls):
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
    def setUpClass(cls):
        p = Process(target=cls._setup)
        p.start()
        p.join()
        p.terminate()
        
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("tests/saved-models/")

    @pytest.mark.skip()
    def test_scheduler_memory_fraction(self):
        m1 = os.path.join(self.folder, self.model1)
        m2 = os.path.join(self.folder, self.model2)
        # Needs to not evaluate to False
        shed = Scheduler(config={'per_process_gpu_memory_fraction': 0.01})
        with self.assertRaises(tf.errors.ResourceExhaustedError):
            pred1a = shed.predict(m1, ["A"])
            raise ValueError("You may need to run this test in isolation")


import unittest
from pathlib import Path
import os

import pytest
import tqdl

from finetune import SequenceLabeler


class TestBackwardsCompatibility(unittest.TestCase):
    model_path = os.path.join("Data", "models", "ner_backwards_compatibility.jl")

    @classmethod
    def setUpClass(cls):
        path = Path(cls.model_path)
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        tqdl.download(
            "https://s3.amazonaws.com/bendropbox/ner_backwards_compatibility.jl",
            str(path),
        )

    def test_model_loads_and_preds(self):
        model = SequenceLabeler.load(self.model_path)
        preds = model.predict(
            [
                "This is a test sequence about Microsoft Corp, Apple Inc and Indico Data Solutions. "
                * 10
            ]
        )
        self.assertTrue(len(preds[0]) == 30)
        acceptable_preds = set(["Microsoft Corp", "Apple Inc", "Indico Data Solutions"])
        for pred in preds[0]:
            self.assertIn(pred["text"], acceptable_preds)


class TestConfig(unittest.TestCase):
    def test_roberta_collapse_whitespace_old_default(self):
        model = SequenceLabeler(collapse_whitespace=True, version="0.8.6")
        assert model.config.collapse_whitespace == True
        model.fit(["test"], [[]])
        model.save("./test_model.jl")
        model = SequenceLabeler.load("./test_model.jl")
        assert model.config.collapse_whitespace == False

    def test_roberta_default_no_change(self):
        model = SequenceLabeler(collapse_whitespace=True)
        assert model.config.collapse_whitespace == True
        model.fit(["test"], [[]])
        model.save("./test_model.jl")
        model = SequenceLabeler.load("./test_model.jl")
        assert model.config.collapse_whitespace == True

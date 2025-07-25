import glob
import os
import unittest
from pathlib import Path

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
        assert len(preds[0]) == 30
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


# TODO: eventually clean this up and push these files to s3.
BUNDLES = glob.glob(os.path.join("/Finetune/tests/backwards_compat_bundles/*.jl"))


@pytest.mark.parametrize("bundle_path", BUNDLES)
def test_backwards_compat_extreme(bundle_path):
    model = SequenceLabeler.load(bundle_path, key="model")
    texts = SequenceLabeler.load(bundle_path, key="texts")
    contexts = SequenceLabeler.load(bundle_path, key="contexts")
    expected_flat_features = SequenceLabeler.load(bundle_path, key="flat_features")
    expected_sequence_features = SequenceLabeler.load(
        bundle_path, key="sequence_features"
    )
    expected_preds = SequenceLabeler.load(bundle_path, key="preds")
    expected_probs = SequenceLabeler.load(bundle_path, key="probs")

    try:
        flat_features = model.featurize(texts, context=contexts)
    except:
        if expected_flat_features is not None:
            raise
        flat_features = None
    assert expected_flat_features is None or flat_features == expected_flat_features

    try:
        sequence_features = model.featurize_sequence(texts, context=contexts)
    except:
        if expected_sequence_features is not None:
            raise
        sequence_features = None
    assert (
        expected_sequence_features is None
        or sequence_features == expected_sequence_features
    )

    try:
        preds = model.predict(texts, context=contexts)
    except:
        if expected_preds is not None:
            raise
        preds = None
    assert expected_preds is None or preds == expected_preds

    try:
        probs = model.predict_proba(texts, context=contexts)
    except:
        if expected_probs is not None:
            raise
        probs = None
    assert expected_probs is None or probs == expected_probs

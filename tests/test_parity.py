import os.path

import numpy as np
import pytest

from finetune import Classifier, SequenceLabeler
from finetune.base_models import BERT, GPT, GPT2, RoBERTa, XDocBase

DIRECTORY = os.path.abspath(os.path.dirname(__file__))

MULTIFIELD_TEST_DATA = [
    [
        "Rick grew up in a troubled household. He never found good support in family, ",
        "and turned to gangs. It wasn't long before Rick got shot in a robbery. The ",
        "incident caused him to turn a new leaf.",
        "He is happy now.",
    ]
]
TEST_DATA = ["this is a test"]


@pytest.mark.xfail
def test_gpt2_featurize():
    # I believe that the issue here is just that expected outputs
    # need re-generating with start and end tokens included.
    model = Classifier(base_model=GPT2)
    np.testing.assert_allclose(
        model.featurize_sequence(TEST_DATA)[0],
        np.load(os.path.join(DIRECTORY, "data/test-gpt2-activations.npy")),
        atol=1e-1,
    )


def test_bert_featurize():
    model = Classifier(base_model=BERT)
    np.testing.assert_allclose(
        model.featurize(TEST_DATA)[0],
        np.load(os.path.join(DIRECTORY, "data/test-bert-activations.npy")),
        atol=1e-1,
    )


def test_roberta_featurize():
    model = Classifier(base_model=RoBERTa)
    np.testing.assert_allclose(
        model.featurize_sequence(TEST_DATA)[0],
        np.load(os.path.join(DIRECTORY, "data/test-roberta-activations.npy"))[0, 1:-1],
        atol=1e-1,
    )


def test_bert_featurize_fp16():
    model = Classifier(base_model=BERT, float_16_predict=True)
    np.testing.assert_allclose(
        model.featurize(TEST_DATA)[0],
        np.load(os.path.join(DIRECTORY, "data/test-bert-activations.npy")),
        atol=1e-1,
    )


def test_roberta_featurize_fp16():
    model = Classifier(base_model=RoBERTa, float_16_predict=True)
    np.testing.assert_allclose(
        model.featurize_sequence(TEST_DATA)[0],
        np.load(os.path.join(DIRECTORY, "data/test-roberta-activations.npy"))[0, 1:-1],
        atol=1e-1,
    )


def test_xdoc_featurize():
    model = SequenceLabeler(base_model=XDocBase)
    np.testing.assert_allclose(
        model.featurize_sequence(
            ["The quick brown fox jumped"],
            context=[[{"left": 0, "top": 0, "right": 0, "bottom": 0, "start": 0, "end": 26}]],
        )[0],
        np.load(os.path.join(DIRECTORY, "data/xdoc_activations.npy"))[0, 1:-1],
        atol=2e-3,
    )

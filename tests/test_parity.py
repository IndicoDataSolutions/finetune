import unittest
import os.path
import pytest

import numpy as np
from tensorflow.data import Dataset

from finetune.encoding.input_encoder import EncodedOutput
from finetune.model import PredictMode
from finetune.base_models import GPT, GPT2, BERT, RoBERTa
from finetune import Classifier, MultiFieldClassifier

DIRECTORY = os.path.abspath(os.path.dirname(__file__))


class TestActivationParity(unittest.TestCase):

    MULTIFIELD_TEST_DATA = [[
        "Rick grew up in a troubled household. He never found good support in family, "
        "and turned to gangs. It wasn't long before Rick got shot in a robbery. The "
        "incident caused him to turn a new leaf.", 
        "He is happy now."
    ]]
    TEST_DATA = ["this is a test"]

    def test_gpt_multifield_parity(self):
        model = MultiFieldClassifier(base_model=GPT)
        np.testing.assert_allclose(
            model.featurize(self.MULTIFIELD_TEST_DATA)[0], 
            np.load(
                os.path.join(
                    DIRECTORY, 
                    'data/test-gpt-multifield-activations.npy'
                )
            ),
            atol=1e-1
        )

    @pytest.mark.xfail
    def test_gpt2_featurize(self):
        # I believe that the issue here is just that expected outputs 
        # need re-generating with start and end tokens included.
        model = Classifier(base_model=GPT2)
        np.testing.assert_allclose(
            model.featurize_sequence(self.TEST_DATA)[0],
            np.load(
                os.path.join(
                    DIRECTORY, 
                    'data/test-gpt2-activations.npy'
                )
            ),
            atol=1e-1
        )

    def test_bert_featurize(self):
        model = Classifier(base_model=BERT)
        np.testing.assert_allclose(
            model.featurize(self.TEST_DATA)[0], 
            np.load(
                os.path.join(
                    DIRECTORY, 
                    'data/test-bert-activations.npy'
                )
            ),
            atol=1e-1
        )

    def test_roberta_featurize(self):
        model = Classifier(base_model=RoBERTa)
        np.testing.assert_allclose(
            model.featurize_sequence(self.TEST_DATA)[0],
            np.load(
                os.path.join(
                    DIRECTORY, 
                    'data/test-roberta-activations.npy'
                )
            )[0, 1:-1],
            atol=1e-1
        )

    def test_bert_featurize_fp16(self):
        model = Classifier(base_model=BERT, float_16_predict=True)
        np.testing.assert_allclose(
            model.featurize(self.TEST_DATA)[0],
            np.load(
                os.path.join(
                    DIRECTORY,
                    'data/test-bert-activations.npy'
                )
            ),
            atol=1e-1
        )
        
    def test_roberta_featurize_fp16(self):
        model = Classifier(base_model=RoBERTa, float_16_predict=True)
        np.testing.assert_allclose(
            model.featurize_sequence(self.TEST_DATA)[0],
            np.load(
                os.path.join(
                    DIRECTORY,
                    'data/test-roberta-activations.npy'
                )
            )[0, 1:-1],
            atol=1e-1
        )
                                         

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import unittest
import spacy
import random
import json
import copy
import shutil

from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from finetune import Classifier, SequenceLabeler, Comparison, ComparisonRegressor, MultipleChoice
from finetune.base_models import TextCNN, BERTModelCased, GPT2Model, GPTModel, RoBERTa, GPT
from finetune.config import get_config
from finetune.util.metrics import (
    sequence_labeling_token_precision,
    sequence_labeling_token_recall,
)
from finetune.datasets.reuters import Reuters
from finetune.encoding.input_encoder import tokenize_context, EncodedOutput


# prevent excessive warning logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class TestAuxiliaryTokenization(unittest.TestCase):
    def test_tokenize_context(self):
        encoded_output = EncodedOutput(
            token_ids=[
                [40478, 40481],
                [ 1180, 40482],
                [  535, 40483],
                [  808, 40484],
                [  289, 40485],
                [17164, 40486],
                [40480, 40487]
            ],
            tokens=[40478, 'everything</w>', "'s</w>", 'only</w>', '$</w>', '80</w>', 40480],
            token_ends=[-1, 10, 12, 17, 19, 21, -1],
            token_starts=[-1, 0, 10, 13, 18, 19, -1],
        )
        context = [
            {'text': "everything's", 'start': 0, 'end': 12, 'left': 10, 'bold': False},
            {'text': "only", 'start': 13, 'end': 17, 'left': 20, 'bold': False},
            {'text': "$80", 'start': 18, 'end': 21, 'left': 30, 'bold': True},
        ]
        config = get_config(**{'default_context': {'left': 0, 'bold': False}})
        expanded_context = tokenize_context(context, encoded_output, config)
        expected = [
            [False, 0],
            [False, 10],
            [False, 10],
            [False, 20],
            [True, 30],
            [True, 30],
            [False, 0]
        ]
        np.testing.assert_array_equal(expected, expanded_context)

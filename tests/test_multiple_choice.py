import os
import unittest
import logging
from copy import copy
from pathlib import Path
import codecs
import json

import tensorflow as tf
# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from finetune import MultipleChoice


class TestQuestionAnswer(unittest.TestCase):

    def test_reasonable_predictions(self):
        model = MultipleChoice(n_epochs=100, val_size=0, max_length=64, batch_size=3, lr=2e-5, lr_schedule="none",
                               val_interval=10000, embed_p_drop=0., attn_p_drop=0., resid_p_drop=0, clf_p_drop=0)
        questions = [
            "Dog, cat, fish, orange, what is the odd one out?",
            "Stocks, Futures, Money, Chicken, what is the odd one out?",
            "England, US, Finland, Penguin, what is the odd one out?",
            "Orange, Yellow, Tiger, Purple, what is the odd one out?",
            "Computer, Cellphone, Coffee, Telephone, what is the odd one out?",
            "Boat, car, chicken, train, what is the odd one out?",
            "Walk, run, william, jump, what is the odd one out?"
        ]

        answers = [
            ["orange", "Dog", "fish", "cat"],
            ["Chicken", "Stocks", "Futures", "Money"],
            ["Penguin", "England", "US", "Finland"],
            ["Tiger", "Orange", "Yellow", "Purple"],
            ["Coffee", "Computer", "Cellphone", "Telephone"],
            ["train", "Boat", "car", "train"],
            ["william", "Walk", "run", "jump"],
        ]

        model.finetune(questions, answers, ["orange", "Chicken", "Penguin", "Tiger", "Coffee", "train", "william"])

        self.assertEqual(["orange"], model.predict(["Dog, cat, fish, orange, what is the odd one out?"],
                                                   [["orange", "Dog", "fish", "cat"]]))

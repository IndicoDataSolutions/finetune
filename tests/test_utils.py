import unittest
import os.path
import random
import json
from collections import Counter

import numpy as np
import tensorflow as tf
import pandas as pd

import finetune
from finetune.encoding.sequence_encoder import indico_to_finetune_sequence, finetune_to_indico_sequence
from finetune.optimizers.gradient_accumulation import get_grad_accumulation_optimizer
from finetune.util.imbalance import compute_class_weights
from finetune.errors import FinetuneError
from finetune import Classifier, SequenceLabeler
from finetune.base_models import GPT, GPT2, BERT
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt2.encoder import GPT2Encoder


class TestFinetuneIndicoConverters(unittest.TestCase):

    def test_invalid_keyword(self):
        with self.assertRaises(FinetuneError):
            model = Classifier(tensorboard='./testing') # should be tensorboard_folder
    
    def test_train_test_tokenization_consistency(self):
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'testdata.csv'))
        df = pd.read_csv(filepath)
        X = []
        Y = []

        for i, row in df.iterrows():
            X.append(row["text"])
            labels = json.loads(row["question_843"])
            for label in labels:
                label['start'] = label['startOffset']
                label['end'] = label['endOffset']
                label['text'] = row["text"][label['start']:label['end']]
            Y.append(labels)

        for multilabel_setting in [True, False]:
            for base_model in [GPT, GPT2, BERT]:
                model = SequenceLabeler(chunk_long_sequences=True, base_model=base_model, multi_label_sequences=multilabel_setting)
                train_encoded = [x for x in model.input_pipeline._text_to_ids(X, Y=Y, pad_token=model.config.pad_token)]
                test_encoded = [x for x in model.input_pipeline._text_to_ids(X)]
                for chunk_id in range(len(train_encoded)):
                    for train_token_ids, test_token_ids in zip(train_encoded[chunk_id].token_ids, test_encoded[chunk_id].token_ids):
                        self.assertEqual(train_token_ids[0], test_token_ids[0])

    def test_whitespace_handling(self):
        # Newline complications
        finetunex = [["Train:", "\n\n\n and test", " tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]
        expectedx = ["Train:\n\n\n and test tokenization must be equivalent"]
        expectedy = [
            [
                {'start': 0, 'end': 18, 'label': "1", 'text': "Train:\n\n\n and test"},
                {'start': 10, 'end': 39, 'label': "2", 'text': "and test tokenization must be"}
            ]
        ]
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)
    
        expectedx = ["Train and test tokenization must be equivalent"]
        expectedy = [
            [
                {'start': 0, 'end': 14, 'label': "1", 'text': "Train and test"},
                {'start': 6, 'end': 35, 'label': "2", 'text': "and test tokenization must be"}
            ]
        ]
    
        # Spaces before labels
        finetunex = [["Train", " and test", " tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]

        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

        # Spaces after labels
        finetunex = [["Train ", "and test ", "tokenization must be ", "equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]
    
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

        # Whitespace anarchy
        finetunex = [["Train", " and test ", "tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]

        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)
        

    def test_overlapping(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico ", "Is the", " best", " hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2",), ("<PAD>",)]
        ]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>",
                                                                 subtoken_predictions=True)

        indicoy = [
            [
                {'start': 0, 'end': 13, 'label': '1', 'text': 'Indico Is the'},
                {'start': 7, 'end': 18, 'label': '2', 'text': 'Is the best'},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred, *_ = indico_to_finetune_sequence(
            raw, indicoy, encoder=encoder, none_value="<PAD>"
        )
        self.assertEqual(finetunex_pred, finetunex)
        self.assertCountEqual(finetuney[0][0], finetuney_pred[0][0])
        self.assertCountEqual(finetuney[0][1], finetuney_pred[0][1])
        self.assertCountEqual(finetuney[0][2], finetuney_pred[0][2])

    def test_overlapping_gpt2(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico", " Is the", " best", " hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2", ), ("<PAD>")]
        ]
        encoder = GPT2Encoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>",
                                                                 subtoken_predictions=False)

        indicoy = [
            [
                {'start': 0, 'end': 13, 'label': '1', 'text': 'Indico Is the'},
                {'start': 7, 'end': 18, 'label': '2', 'text': 'Is the best'},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred, *_ = indico_to_finetune_sequence(
            raw, indicoy, encoder=encoder, none_value="<PAD>"
        )
        self.assertEqual(finetunex_pred, finetunex)
        self.assertCountEqual(finetuney[0][0], finetuney_pred[0][0])
        self.assertCountEqual(finetuney[0][1], finetuney_pred[0][1])
        self.assertCountEqual(finetuney[0][2], finetuney_pred[0][2])

    def test_overlapping_gpt2_subtokens(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico", " Is the", " best", " hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2", ), ("<PAD>")]
        ]
        encoder = GPT2Encoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>",
                                                                 subtoken_predictions=True)

        indicoy = [
            [
                {'start': 0, 'end': 13, 'label': '1', 'text': 'Indico Is the'},
                {'start': 6, 'end': 18, 'label': '2', 'text': ' Is the best'},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred, *_ = indico_to_finetune_sequence(
            raw, indicoy, encoder=encoder, none_value="<PAD>"
        )
        self.assertEqual(finetunex_pred, finetunex)
        self.assertCountEqual(finetuney[0][0], finetuney_pred[0][0])
        self.assertCountEqual(finetuney[0][1], finetuney_pred[0][1])
        self.assertCountEqual(finetuney[0][2], finetuney_pred[0][2])

    def test_nested_labels(self):
        raw = ["Indico Is the best"]
        finetunex = [
            ["Indico ", "Is the", " best"]
        ]
        finetuney = [
            [("1", ), ("1", "2", "3"), ("1", )]
        ]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>")

        indicoy = [
            [
                {'start': 0, 'end': 18, 'label': '1', 'text': 'Indico Is the best'},
                {'start': 7, 'end': 13, 'label': '2', 'text': 'Is the'},
                {'start': 7, 'end': 13, 'label': '3', 'text': 'Is the'}
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred, *_ = indico_to_finetune_sequence(
            raw, indicoy, encoder=encoder, none_value="<PAD>"
        )
        self.assertEqual(finetunex_pred, finetunex)
        self.assertCountEqual(finetuney[0][0], finetuney_pred[0][0])
        self.assertCountEqual(finetuney[0][1], finetuney_pred[0][1])
        self.assertCountEqual(finetuney[0][2], finetuney_pred[0][2])

    def test_overlapping_labels_with_single_label(self):
        text = ["Indico Rules"]
        finetunex = [
            ["Indic", "o", " Rules"]
        ]
        finetuney = [
            ["1", "1", "2"]
        ]
        indicoy = [
            [
                {'start': 0, 'end': 6, 'label': '1', 'text': 'Indico'},
                {'start': 5, 'end': 12, 'label': '2', 'text': 'o Rules'},
            ]
        ]

        finetunex_pred, finetuney_pred, *_ = indico_to_finetune_sequence(
            text, indicoy, encoder=GPTEncoder(), none_value="<PAD>", multi_label=False
        )
        self.assertEqual(finetunex_pred, finetunex)
        self.assertCountEqual(finetuney[0][0], finetuney_pred[0][0])
        self.assertCountEqual(finetuney[0][1], finetuney_pred[0][1])
        self.assertCountEqual(finetuney[0][2], finetuney_pred[0][2])

    def test_three_overlapping_labels(self):
        raw = ["Indico Is the very best"]
        finetunex = [
            ["Indico ", "Is the very", " best"]
        ]
        finetuney = [
            [("<PAD>", ), ("1", "2", "3"), ("1", "3")]
        ]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>")
        indicoy_pred = [sorted(seq, key=lambda x: x['label']) for seq in indicoy_pred]
        indicoy = [
            sorted(
                [
                    {'start': 7, 'end': 18, 'label': '2', 'text': 'Is the very'},
                    {'start': 7, 'end': 23, 'label': '1', 'text': 'Is the very best'},
                    {'start': 7, 'end': 23, 'label': '3', 'text': 'Is the very best'}
                ],
                key=lambda x: x['label']
            )
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred, *_ = indico_to_finetune_sequence(
            raw, indicoy, encoder=encoder, none_value="<PAD>"
        )
        self.assertEqual(finetunex_pred, finetunex)
        self.assertCountEqual(finetuney[0][0], finetuney_pred[0][0])
        self.assertCountEqual(finetuney[0][1], finetuney_pred[0][1])
        self.assertCountEqual(finetuney[0][2], finetuney_pred[0][2])

    def test_compute_class_weights(self):
        # regression test for issue #181
        np.random.seed(0)
        y = np.random.choice(a=[0, 1, 2], size=1000, p=[0.3, 0.6, 0.1])
        class_counts = Counter(y)
        weights = compute_class_weights('log', class_counts=class_counts)
        self.assertEqual(weights[1], 1.0)


class TestGradientAccumulation(unittest.TestCase):

    def test_gradient_accumulating_optimizer(self):
        loss = tf.get_variable("loss", shape=1)
        lr = 0.1
        opt = get_grad_accumulation_optimizer(tf.train.GradientDescentOptimizer, 2)(lr)
        train_op = opt.minimize(tf.abs(loss))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            val_before = sess.run(loss)
            grad_before = np.sign(val_before)
            sess.run(train_op)

            val_after1 = sess.run(loss)
            grad_after1 = np.sign(val_after1)
            sess.run(train_op)

            val_after2 = sess.run(loss)

            self.assertEqual(val_before - (grad_before + grad_after1) * lr, val_after2)  # check 2 steps of update have been made.
            self.assertEqual(val_before, val_after1)  # first step should not actually do anything


if __name__ == '__main__':
    unittest.main()

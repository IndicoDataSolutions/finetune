import unittest

import numpy as np
import tensorflow as tf

import finetune
from finetune.encoding.sequence_encoder import indico_to_finetune_sequence, finetune_to_indico_sequence
from finetune.optimizers.gradient_accumulation import get_grad_accumulation_optimizer
from finetune.util.imbalance import compute_class_weights
from finetune.errors import FinetuneError
from finetune import Classifier
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt2.encoder import GPT2Encoder


class TestFinetuneIndicoConverters(unittest.TestCase):

    def test_invalid_keyword(self):
        with self.assertRaises(FinetuneError):
            model = Classifier(tensorboard='./testing') # should be tensorboard_folder

    def test_overlapping(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico ", "Is the", " best", " hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2", ), ("<PAD>")]
        ]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, encoder=encoder, none_value="<PAD>"
        )

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
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, encoder=encoder, none_value="<PAD>"
        )

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
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, encoder=encoder, none_value="<PAD>"
        )

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
        print(finetuney)
        print(finetuney_pred)
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
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, encoder=encoder, none_value="<PAD>"
        )
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
        weights = compute_class_weights('log', y)
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

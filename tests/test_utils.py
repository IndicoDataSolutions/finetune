import unittest

import numpy as np

from finetune.utils import indico_to_finetune_sequence, finetune_to_indico_sequence
from finetune.imbalance import compute_class_weights


class TestFinetuneIndicoConverters(unittest.TestCase):

    def test_overlapping(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico ", "Is the", " best", " hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2", ), ("<PAD>")]
        ]
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney)

        indicoy = [
            [
                {'start': 0, 'end': 13, 'label': '1', 'text': 'Indico Is the'},
                {'start': 7, 'end': 18, 'label': '2', 'text': 'Is the best'},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred = indico_to_finetune_sequence(raw, indicoy)
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
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney)

        indicoy = [
            [
                {'start': 0, 'end': 18, 'label': '1', 'text': 'Indico Is the best'},
                {'start': 7, 'end': 13, 'label': '2', 'text': 'Is the'},
                {'start': 7, 'end': 13, 'label': '3', 'text': 'Is the'}
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred = indico_to_finetune_sequence(raw, indicoy)
        self.assertEqual(finetunex_pred, finetunex)
        self.assertCountEqual(finetuney[0][0], finetuney_pred[0][0])
        self.assertCountEqual(finetuney[0][1], finetuney_pred[0][1])
        self.assertCountEqual(finetuney[0][2], finetuney_pred[0][2])


    def test_three_overlapping_labels(self):
        raw = ["Indico Is the best"]
        finetunex = [
            ["Indico ", "Is the", " best"]
        ]
        finetuney = [
            [("<PAD>", ), ("1", "2", "3"), ("1", "3")]
        ]
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney)
        indicoy = [
            [
                {'start': 7, 'end': 13, 'label': '2', 'text': 'Is the'},
                {'start': 7, 'end': 18, 'label': '1', 'text': 'Is the best'},
                {'start': 7, 'end': 18, 'label': '3', 'text': 'Is the best'}
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

        finetunex_pred, finetuney_pred = indico_to_finetune_sequence(raw, indicoy)
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

 
if __name__ == '__main__':
    unittest.main()

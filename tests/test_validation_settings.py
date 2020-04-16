import unittest

from finetune.util.input_utils import validation_settings

class TestValidationSettings(unittest.TestCase):

    def test_validation_settings(self):
        """
        Ensure LM only training does not error out
        """
        val_size, val_interval = validation_settings(dataset_size=30, batch_size=4, val_size=0, val_interval=None, keep_best_model=False)
        self.assertEqual(val_size, 0)

        val_size, val_interval = validation_settings(dataset_size=80, batch_size=4, val_size=0.05, val_interval=None, keep_best_model=False)
        self.assertEqual(val_size, 4)
        self.assertEqual(val_interval, 4)

        val_size, val_interval = validation_settings(dataset_size=80, batch_size=2, val_size=0.05, val_interval=None, keep_best_model=False)
        self.assertEqual(val_size, 4)
        self.assertEqual(val_interval, 8)

        val_size, val_interval = validation_settings(dataset_size=400, batch_size=4, val_size=0.05, val_interval=None, keep_best_model=False)
        self.assertEqual(val_size, 20)
        self.assertEqual(val_interval, 20)

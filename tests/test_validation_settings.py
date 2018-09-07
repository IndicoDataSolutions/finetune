import unittest

from finetune import Classifier


class TestValidationSettings(unittest.TestCase):

    def test_validation_settings(self):
        """
        Ensure LM only training does not error out
        """
        model = Classifier()


        val_size, val_interval = model.validation_settings(n_examples=30, batch_size=4)
        self.assertEqual(val_size, 0)

        val_size, val_interval = model.validation_settings(n_examples=80, batch_size=4)
        self.assertEqual(val_size, 5)
        self.assertEqual(val_interval, 20)

        val_size, val_interval = model.validation_settings(n_examples=1000, batch_size=4)
        self.assertEqual(val_size, 50)
        self.assertEqual(val_interval, 130)

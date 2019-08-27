import unittest

from finetune import Classifier


class TestValidationSettings(unittest.TestCase):

    def test_validation_settings(self):
        """
        Ensure LM only training does not error out
        """
        model = Classifier()

        val_size, val_interval = model.input_pipeline.validation_settings(n_examples=30, batch_size=4)
        self.assertEqual(val_size, 0)

        model = Classifier(val_size=0.05)
        val_size, val_interval = model.input_pipeline.validation_settings(n_examples=80, batch_size=4)
        self.assertEqual(val_size, 4)
        self.assertEqual(val_interval, 4)

        model = Classifier(val_size=0.05)
        val_size, val_interval = model.input_pipeline.validation_settings(n_examples=80, batch_size=2)
        self.assertEqual(val_size, 4)
        self.assertEqual(val_interval, 8)

        val_size, val_interval = model.input_pipeline.validation_settings(n_examples=400, batch_size=4)
        self.assertEqual(val_size, 20)
        self.assertEqual(val_interval, 20)

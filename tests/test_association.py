import os
import unittest

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest

from sklearn.model_selection import train_test_split
from finetune import Association

from finetune.config import get_config
from finetune.datasets.treebank_association import TreebankNounVP

SKIP_LM_TESTS = get_config().base_model.is_bidirectional


class TestAssociation(unittest.TestCase):

    @classmethod
    def get_data_and_schema(cls, rows=None):
        data = TreebankNounVP()
        return (
            data.get_data(rows),
            {
                "association_types": ["has_verb"],
                "viable_edges": {
                    "noun_phrase": [
                        ["verb", "has_verb"],
                    ]
                    , "verb": [None]
                }
            }
        )

    def setUp(self):
        data, schema = self.get_data_and_schema(20)
        self.texts, self.labels = data
        self.model = Association(batch_size=2, max_length=32, **schema)

    @pytest.mark.skipif(SKIP_LM_TESTS, reason="Bidirectional models do not yet support LM functions")
    def test_fit_lm_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(self.texts, self.labels, test_size=0.1)
        self.model.fit(train_texts)
        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(test_texts))
        self.assertIsInstance(predictions[0], list)
        self.assertGreater(sum(len(p) for p in predictions), 0)
        # make sure there is at least one prediction, if there isn't somethings probably broken.

        for pred in predictions:
            pred_has_verb = any(p_i["label"] == "verb" for p_i in pred)
            for p_i in pred:
                self.assertIsInstance(p_i, dict)
                self.assertIsInstance(p_i["label"], str)
                self.assertIn(p_i["label"], ["verb", "noun_phrase"])

                if p_i["label"] == "noun_phrase" and pred_has_verb:
                    self.assertIn("associations", p_i)
                    # a noun_phrase MUST have a verb by the above schema.

                    self.assertEqual(p_i["associations"]["relationship"], "has_verb")
                    # has_verb is the only viable edge for a noun_phrase.

                    self.assertIn("prob", p_i["associations"])
                    self.assertTrue(0.0 < p_i["associations"]["prob"] < 1.0)
                    self.assertEqual(pred[p_i["associations"]["index"]]["label"], "verb")
                    # check it is associated with a verb
                else:
                    self.assertNotIn("associations", p_i)
                    # Verbs cannot have any associations by the above schema

            self.model.predict(test_texts)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            self.texts, self.labels, test_size=0.1
        )
        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(test_texts))
        self.assertIsInstance(predictions[0], list)
        self.assertGreater(sum(len(p) for p in predictions), 0)
        # make sure there is at least one prediction, if there isn't somethings probably broken.

        for pred in predictions:
            pred_has_verb = any(p_i["label"] == "verb" for p_i in pred)
            for p_i in pred:
                self.assertIsInstance(p_i, dict)
                self.assertIsInstance(p_i["label"], str)
                self.assertIn(p_i["label"], ["verb", "noun_phrase"])

                if p_i["label"] == "noun_phrase" and pred_has_verb:
                    self.assertIn("associations", p_i)
                    # a noun_phrase MUST have a verb by the above schema.

                    self.assertEqual(p_i["associations"]["relationship"], "has_verb")
                    # has_verb is the only viable edge for a noun_phrase.

                    self.assertIn("prob", p_i["associations"])
                    self.assertTrue(0.0 < p_i["associations"]["prob"] < 1.0)
                    self.assertEqual(pred[p_i["associations"]["index"]]["label"], "verb")
                    # check it is associated with a verb
                else:
                    self.assertNotIn("associations", p_i)
                    # Verbs cannot have any associations by the above schema

            self.model.predict(test_texts)


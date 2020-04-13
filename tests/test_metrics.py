import unittest
from finetune.util.metrics import (
    seq_recall,
    seq_precision,
    get_seq_count_fn,
    micro_f1,
    model_f1,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        x = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
        y_true = [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ]
        y_overlap = [
            {"start": 5, "end": 16, "label": "entity"},
            {"start": 34, "end": 50, "label": "date"},
        ]
        y_extra_overlap = [
            {"start": 6, "end": 10, "label": "entity"},
            {"start": 15, "end": 23, "label": "entity"},
            {"start": 34, "end": 50, "label": "date"},
        ]
        y_superset = [
            {"start": 6, "end": 21, "label": "entity"},
            {"start": 38, "end": 60, "label": "date"},
        ]
        y_false_pos = [
            {"start": 21, "end": 28, "label": "entity"},
            {"start": 62, "end": 65, "label": "date"},
        ]
        y_false_neg = [
            {"start": 7, "end": 20, "label": "entity"},
        ]

        self.X = [x for i in range(10)]
        self.Y_false_pos = self.extend_label(self.X, y_false_pos, 10)
        self.Y_true = self.extend_label(self.X, y_true, 10)
        self.Y_overlap = self.extend_label(self.X, y_overlap, 10)
        self.Y_extra_overlap = self.extend_label(self.X, y_extra_overlap, 10)
        self.Y_superset = self.extend_label(self.X, y_superset, 10)
        self.Y_false_neg = self.extend_label(self.X, y_false_neg, 10)

        self.seq_expected_correct = {
            "entity": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 10,
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
            },
            "date": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 10,
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
            },
            "micro-f1": 1.0,
            "weighted-f1": 1.0,
            "macro-f1": 1.0,
        }
        self.seq_expected_incorrect = {
            "entity": {
                "false_positives": 10,
                "false_negatives": 10,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
            },
            "date": {
                "false_positives": 10,
                "false_negatives": 10,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
            },
            "micro-f1": 0.0,
            "macro-f1": 0.0,
            "weighted-f1": 0.0,
        }

    def remove_label(self, recs, label):
        return [
            [pred for pred in rec if not pred.get("label") == label] for rec in recs
        ]

    def insert_text(self, docs, labels):
        if len(docs) != len(labels):
            raise ValueError(
                "Number of documents must be equal to the number of labels"
            )
        for doc, label in zip(docs, labels):
            for l in label:
                l["text"] = doc[l["start"] : l["end"]]
        return labels

    def extend_label(self, text, label, amt):
        return self.insert_text(text, [label for _ in range(amt)])

    def check_metrics(self, Y, Y_pred, expected, span_type=None):
        counts = get_seq_count_fn(span_type)(Y, Y_pred)
        precisions = seq_precision(Y, Y_pred, span_type=span_type)
        recalls = seq_recall(Y, Y_pred, span_type=span_type)
        micro_f1_score = model_f1(Y, Y_pred, span_type=span_type, average="micro")
        per_class_f1s = model_f1(Y, Y_pred, span_type=span_type)
        weighted_f1 = model_f1(Y, Y_pred, span_type=span_type, average="weighted")
        macro_f1 = model_f1(Y, Y_pred, span_type=span_type, average="macro")
        for cls_ in counts:
            for metric in counts[cls_]:
                self.assertEqual(len(counts[cls_][metric]), expected[cls_][metric])
            self.assertAlmostEqual(recalls[cls_], expected[cls_]["recall"], places=3)
            self.assertAlmostEqual(
                per_class_f1s[cls_]["f1-score"], expected[cls_]["f1-score"], places=3
            )
            self.assertAlmostEqual(
                precisions[cls_], expected[cls_]["precision"], places=3
            )
        self.assertAlmostEqual(micro_f1_score, expected["micro-f1"], places=3)
        self.assertAlmostEqual(weighted_f1, expected["weighted-f1"], places=3)
        self.assertAlmostEqual(macro_f1, expected["macro-f1"], places=3)

    def test_token_incorrect(self):
        expected = {
            "entity": {
                "false_positives": 10,
                "false_negatives": 20,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
            },
            "date": {
                "false_positives": 10,
                "false_negatives": 40,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
            },
            "micro-f1": 0.0,
            "macro-f1": 0.0,
            "weighted-f1": 0.0,
        }
        self.check_metrics(self.Y_true, self.Y_false_pos, expected, span_type="token")

    def test_token_correct(self):
        expected = {
            "entity": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 20,
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
            },
            "date": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 40,
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
            },
            "micro-f1": 1.0,
            "macro-f1": 1.0,
            "weighted-f1": 1.0,
        }

        self.check_metrics(self.Y_true, self.Y_true, expected, span_type="token")

    def test_token_mixed(self):
        Y_mixed = self.Y_false_pos[:5] + self.Y_true[:5]
        expected = {
            "entity": {
                "false_positives": 5,
                "false_negatives": 10,
                "correct": 10,
                "precision": 0.66666,
                "recall": 0.5,
                "f1-score": 0.571,
            },
            "date": {
                "false_positives": 5,
                "false_negatives": 20,
                "correct": 20,
                "precision": 0.8,
                "recall": 0.5,
                "f1-score": 0.6153,
            },
            "micro-f1": 0.6,
            "macro-f1": 0.593,
            "weighted-f1": 0.601,
        }

        self.check_metrics(
            self.Y_true, Y_mixed, expected, span_type="token",
        )

        # Move some predictions from correct to false negative
        Y_mixed_false_negs = (
            self.Y_false_pos[:5] + self.Y_true[:2] + self.Y_false_neg[:3]
        )
        expected["date"] = {
            "false_positives": 5,
            "false_negatives": 32,
            "correct": 8,
            "precision": 0.615,
            "recall": 0.2,
            "f1-score": 0.302,
        }
        expected["micro-f1"] = 0.409
        expected["macro-f1"] = 0.437
        expected["weighted-f1"] = 0.392
        self.check_metrics(
            self.Y_true, Y_mixed_false_negs, expected, span_type="token",
        )

    def test_seq_correct(self):

        # Overlaps
        for y_set in [self.Y_true, self.Y_overlap, self.Y_extra_overlap]:
            self.check_metrics(
                self.Y_true, y_set, self.seq_expected_correct, span_type="overlap"
            )

        # Exact
        self.check_metrics(
            self.Y_true, self.Y_true, self.seq_expected_correct, span_type="exact"
        )

        # Superset
        self.check_metrics(
            self.Y_true,
            self.Y_superset,
            self.seq_expected_correct,
            span_type="superset",
        )

    def test_seq_incorrect(self):
        # Overlap
        self.check_metrics(
            self.Y_true,
            self.Y_false_pos,
            self.seq_expected_incorrect,
            span_type="overlap",
        )

        # Exact
        self.check_metrics(
            self.Y_true,
            self.Y_false_pos,
            self.seq_expected_incorrect,
            span_type="exact",
        )

        # Superset
        self.check_metrics(
            self.Y_true,
            self.Y_false_pos,
            self.seq_expected_incorrect,
            span_type="superset",
        )

    def test_seq_mixed(self):
        expected = {
            "entity": {
                "false_positives": 4,
                "false_negatives": 4,
                "correct": 6,
                "precision": 0.6,
                "recall": 0.6,
                "f1-score": 0.6,
            },
            "date": {
                "false_positives": 4,
                "false_negatives": 4,
                "correct": 6,
                "precision": 0.6,
                "recall": 0.6,
                "f1-score": 0.6,
            },
            "micro-f1": 0.6,
            "macro-f1": 0.6,
            "weighted-f1": 0.6,
        }
        Y_mixed_overlap = self.Y_false_pos[:4] + self.Y_overlap[:6]
        Y_mixed_extra_overlap = self.Y_false_pos[:4] + self.Y_extra_overlap[:6]
        Y_mixed_exact = self.Y_false_pos[:4] + self.Y_true[:6]
        for y_set in [Y_mixed_overlap, Y_mixed_extra_overlap, Y_mixed_exact]:
            self.check_metrics(
                self.Y_true, y_set, expected=expected, span_type="overlap"
            )

        # Move 3 predictions from correct to false negative
        Y_mixed_overlap_false_negs = (
            self.Y_false_pos[:4] + self.Y_overlap[:3] + self.Y_false_neg[:3]
        )
        expected["date"] = {
            "false_positives": 4,
            "false_negatives": 7,
            "correct": 3,
            "precision": 0.42857,
            "recall": 0.3,
            "f1-score": 0.353,
        }
        expected["micro-f1"] = 0.4864
        expected["weighted-f1"] = 0.476
        expected["macro-f1"] = 0.476
        self.check_metrics(
            self.Y_true,
            Y_mixed_overlap_false_negs,
            expected=expected,
            span_type="overlap",
        )
        Y_mixed_exact_false_negs = (
            self.Y_false_pos[:4] + self.Y_true[:3] + self.Y_false_neg[:3]
        )
        self.check_metrics(
            self.Y_true, Y_mixed_exact_false_negs, expected=expected, span_type="exact"
        )
        Y_mixed_superset_false_negs = (
            self.Y_false_pos[:4] + self.Y_superset[:3] + self.Y_false_neg[:3]
        )
        self.check_metrics(
            self.Y_true,
            Y_mixed_superset_false_negs,
            expected=expected,
            span_type="superset",
        )

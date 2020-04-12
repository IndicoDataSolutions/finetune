import unittest
from finetune.util.metrics import (
    sequence_labeling_token_precision,
    sequence_labeling_token_recall,
    sequence_labeling_micro_token_f1,
    sequence_labeling_overlap_precision,
    sequence_labeling_overlap_recall,
    sequence_labeling_token_counts,
    sequence_labeling_overlaps,
    sequence_labeling_micro_f1,
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

    def check_metrics(
        self,
        Y,
        Y_pred,
        expected,
        count_fn=None,
        precision_fn=None,
        recall_fn=None,
        f1_fn=None,
    ):
        counts = count_fn(Y, Y_pred)
        precisions = precision_fn(Y, Y_pred)
        recalls = recall_fn(Y, Y_pred)
        print("precisions", precisions, "recalls", recalls)
        for cls_ in counts:
            for metric in counts[cls_]:
                self.assertAlmostEqual(
                    len(counts[cls_][metric]), expected[cls_][metric], places=3
                )
            self.assertAlmostEqual(
                precisions[cls_], expected[cls_]["precision"], places=3
            )
            self.assertAlmostEqual(recalls[cls_], expected[cls_]["recall"], places=3)
        if f1_fn:
            micro_f1 = f1_fn(Y, Y_pred)
            self.assertAlmostEqual(micro_f1, expected["micro-f1"], places=3)

    def test_token_incorrect(self):
        expected = {
            "entity": {
                "false_positives": 10,
                "false_negatives": 20,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
            },
            "date": {
                "false_positives": 10,
                "false_negatives": 40,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
            },
            "micro-f1": 0.0,
        }

        self.check_metrics(
            self.Y_true,
            self.Y_false_pos,
            expected,
            count_fn=sequence_labeling_token_counts,
            precision_fn=sequence_labeling_token_precision,
            recall_fn=sequence_labeling_token_recall,
            f1_fn=sequence_labeling_micro_token_f1,
        )

    def test_token_correct(self):
        expected = {
            "entity": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 20,
                "precision": 1.0,
                "recall": 1.0,
            },
            "date": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 40,
                "precision": 1.0,
                "recall": 1.0,
            },
            "micro-f1": 1.0,
        }
        self.check_metrics(
            self.Y_true,
            self.Y_true,
            expected,
            count_fn=sequence_labeling_token_counts,
            precision_fn=sequence_labeling_token_precision,
            recall_fn=sequence_labeling_token_recall,
            f1_fn=sequence_labeling_micro_token_f1,
        )

    def test_token_mixed(self):
        Y_mixed = self.Y_false_pos[:5] + self.Y_true[:5]
        expected = {
            "entity": {
                "false_positives": 5,
                "false_negatives": 10,
                "correct": 10,
                "precision": 0.66666,
                "recall": 0.5,
            },
            "date": {
                "false_positives": 5,
                "false_negatives": 20,
                "correct": 20,
                "precision": 0.8,
                "recall": 0.5,
            },
            "micro-f1": 0.6,
        }
        self.check_metrics(
            self.Y_true,
            Y_mixed,
            expected,
            count_fn=sequence_labeling_token_counts,
            precision_fn=sequence_labeling_token_precision,
            recall_fn=sequence_labeling_token_recall,
            f1_fn=sequence_labeling_micro_token_f1,
        )

    def test_seq_overlap_correct(self):
        expected = {
            "entity": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 10,
                "precision": 1.0,
                "recall": 1.0,
            },
            "date": {
                "false_positives": 0,
                "false_negatives": 0,
                "correct": 10,
                "precision": 1.0,
                "recall": 1.0,
            },
            "micro-f1": 1.0,
        }
        self.check_metrics(
            self.Y_true,
            self.Y_true,
            expected,
            count_fn=sequence_labeling_overlaps,
            precision_fn=sequence_labeling_overlap_precision,
            recall_fn=sequence_labeling_token_recall,
            f1_fn=sequence_labeling_micro_f1,
        )
        self.check_metrics(
            self.Y_true,
            self.Y_overlap,
            expected,
            count_fn=sequence_labeling_overlaps,
            precision_fn=sequence_labeling_overlap_precision,
            recall_fn=sequence_labeling_overlap_recall,
        )
        self.check_metrics(
            self.Y_true,
            self.Y_extra_overlap,
            expected,
            count_fn=sequence_labeling_overlaps,
            precision_fn=sequence_labeling_overlap_precision,
            recall_fn=sequence_labeling_overlap_recall,
            f1_fn=sequence_labeling_micro_f1,
        )

    def test_seq_overlap_incorrect(self):
        expected = {
            "entity": {
                "false_positives": 10,
                "false_negatives": 10,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
            },
            "date": {
                "false_positives": 10,
                "false_negatives": 10,
                "correct": 0,
                "precision": 0.0,
                "recall": 0.0,
            },
            "micro-f1": 1.0,
        }

        self.check_metrics(
            self.Y_true,
            self.Y_false_pos,
            expected,
            count_fn=sequence_labeling_overlaps,
            precision_fn=sequence_labeling_overlap_precision,
            recall_fn=sequence_labeling_overlap_recall,
        )

    def test_seq_overlap_mixed(self):
        expected = {
            "entity": {
                "false_positives": 4,
                "false_negatives": 4,
                "correct": 6,
                "precision": 0.6,
                "recall": 0.6,
            },
            "date": {
                "false_positives": 4,
                "false_negatives": 4,
                "correct": 6,
                "precision": 0.6,
                "recall": 0.6,
            },
            "micro-f1": 0.6,
        }
        Y_mixed_overlap = self.Y_false_pos[:4] + self.Y_overlap[:6]
        Y_mixed_extra_overlap = self.Y_false_pos[:4] + self.Y_extra_overlap[:6]
        Y_mixed_exact = self.Y_false_pos[:4] + self.Y_true[:6]
        for y_set in [Y_mixed_overlap, Y_mixed_extra_overlap, Y_mixed_exact]:
            self.check_metrics(
                self.Y_true,
                y_set,
                expected=expected,
                count_fn=sequence_labeling_overlaps,
                precision_fn=sequence_labeling_overlap_precision,
                recall_fn=sequence_labeling_overlap_recall,
                f1_fn=sequence_labeling_micro_f1,
            )

        # Add some records with no predicted date
        Y_mixed_no_date = self.Y_false_pos[:4] + self.Y_true[:3] + self.Y_false_neg[:3]
        expected["date"] = {
            "false_positives": 4,
            "false_negatives": 7,
            "correct": 3,
            "precision": 0.42857,
            "recall": 0.3,
        }
        expected["micro-f1"] = 0.4864
        self.check_metrics(
            self.Y_true,
            Y_mixed_no_date,
            expected=expected,
            count_fn=sequence_labeling_overlaps,
            precision_fn=sequence_labeling_overlap_precision,
            recall_fn=sequence_labeling_overlap_recall,
            f1_fn=sequence_labeling_micro_f1,
        )

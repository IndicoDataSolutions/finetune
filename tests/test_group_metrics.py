import unittest
from copy import deepcopy
from finetune.util.group_metrics import (
    get_count_fn,
    calc_group_assignment,
    create_joint_groups,
    get_metrics,
    group_metrics,
    joint_metrics,
)

class TestGroupMetrics(unittest.TestCase):
    def setUp(self):
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        self.base_groups = [
            {
                "spans": [
                    {"start": 0, "end": 17, "text": "five percent (5%)"},
                    {"start": 61, "end": 78, "text": "nine percent (9%)"},
                ],
                "label": "class1"
            },
            {
                "spans": [
                    {"start": 42, "end": 58, "text": "two percent (2%)"},
                ],
                "label": "class1"
            },
            {
                "spans": [
                    {"start": 81, "end": 99, "text": "three percent (3%)"},
                ],
                "label": "class2"
            }
        ]
        self.incorrect_groups = [
            {
                "spans": [
                    {"start": 0, "end": 17, "text": "five percent (5%)"},
                ],
                "label": "class1"
            },
        ]
        # Overpredict one class
        self.overpredict_groups = self.base_groups + self.incorrect_groups
        # Underpredict one class
        self.underpredict_groups = self.base_groups[1:]
        # Overpredict one class, underpredict the other
        self.under_over_predict_groups = deepcopy(self.overpredict_groups)
        self.under_over_predict_groups[1]["label"] = "class2"
        self.under_over_predict_groups[3]["label"] = "class2"

        self.incorrect_label_groups = deepcopy(self.base_groups)
        for group in self.incorrect_label_groups:
            if group["label"] == "class1":
                group["label"] = "class2"
            else:
                group["label"] = "class1"
        self.new_label_groups = deepcopy(self.base_groups)
        for group in self.new_label_groups:
            group["label"] = "class3"

        self.correct_expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 1,
                    "recall": 1,
                    "f1-score": 1,
                    "support": 20,
                    "true_positives": 20,
                    "false_positives": 0,
                    "false_negatives": 0,
                },
                "class2": {
                    "precision": 1,
                    "recall": 1,
                    "f1-score": 1,
                    "support": 10,
                    "true_positives": 10,
                    "false_positives": 0,
                    "false_negatives": 0,
                },
            },
            "avg_f1": {
                "micro": 1,
                "macro": 1,
                "weighted": 1,
            }
        }

        self.entities = [
            # 0, Match span exactly, in group
            {"start": 0, "end": 17, "label": "a"},
            # 1, Overlap span
            {"start": 70, "end": 85, "label": "b"},
            # 2, Supserset span
            {"start": 38, "end": 62, "label": "a"},
            # 3, Subset span, in group
            {"start": 44, "end": 54, "label": "b"},
            # 4, Not in span
            {"start": 18, "end": 30, "label": "a"},
        ]
        self.joint_base_groups = deepcopy(self.base_groups)
        self.joint_base_groups[0]["entities"] = [self.entities[0]]
        self.joint_base_groups[1]["entities"] = [self.entities[3]]
        self.joint_base_groups[2]["entities"] = []


    def test_group_assignment(self):
        # Simple tests, since the heavy lifting is done by the count fns
        
        count_fn = get_count_fn("group", "exact")

        correct_idxs = [(0, 0), (1, 1), (2, 2)]
        correct_counts = [
            [(1, 0, 0), (0, 1, 1), (0, 1, 1)],
            [(0, 1, 1), (1, 0, 0), (0, 1, 1)],
            [(0, 1, 1), (0, 1, 1), (1, 0, 0)],
        ]
        self.check_group_assignment(
            count_fn, self.base_groups, self.base_groups,
            correct_idxs, correct_counts
        )

        correct_idxs = [(0, 2), (1, 1), (2, 0)]
        correct_counts = correct_counts[::-1]
        self.check_group_assignment(
            count_fn, self.base_groups, self.base_groups[::-1],
            correct_idxs, correct_counts
        )
        # Assign idxs are sorted by pred idx, so correct values are the same
        self.check_group_assignment(
            count_fn, self.base_groups[::-1], self.base_groups,
            correct_idxs, correct_counts
        )

    def test_group_assignment_rectangle(self):
        count_fn = get_count_fn("group", "exact")
        groups_a = self.base_groups
        groups_b = self.base_groups[:2]

        # Normal
        correct_idxs = [(0, 0), (1, 1)]
        correct_counts = [
            [(1, 0, 0), (0, 1, 1)], 
            [(0, 1, 1), (1, 0, 0)], 
            [(0, 1, 1), (0, 1, 1)],
        ]
        self.check_group_assignment(
            count_fn, groups_a, groups_b,
            correct_idxs, correct_counts
        )

        correct_counts = [
            [(1, 0, 0), (0, 1, 1), (0, 1, 1)], 
            [(0, 1, 1), (1, 0, 0), (0, 1, 1)], 
        ]
        self.check_group_assignment(
            count_fn, groups_b, groups_a,
            correct_idxs, correct_counts
        )

        # Group a inverted
        correct_idxs = [(1, 1), (2, 0)]
        correct_counts = [
            [(0, 1, 1), (0, 1, 1)],
            [(0, 1, 1), (1, 0, 0)], 
            [(1, 0, 0), (0, 1, 1)], 
        ]
        self.check_group_assignment(
            count_fn, groups_a[::-1], groups_b,
            correct_idxs, correct_counts
        )

        correct_idxs = [(0, 2), (1, 1)]
        correct_counts = [
            [(0, 1, 1), (0, 1, 1), (1, 0, 0)],
            [(0, 1, 1), (1, 0, 0), (0, 1, 1)], 
        ]
        self.check_group_assignment(
            count_fn, groups_b, groups_a[::-1],
            correct_idxs, correct_counts
        )

        # Group b inverted
        correct_idxs = [(0, 1), (1, 0)]
        correct_counts = [
            [(0, 1, 1), (1, 0, 0)], 
            [(1, 0, 0), (0, 1, 1)], 
            [(0, 1, 1), (0, 1, 1)],
        ]
        self.check_group_assignment(
            count_fn, groups_a, groups_b[::-1],
            correct_idxs, correct_counts
        )

        correct_idxs = [(0, 1), (1, 0)]
        correct_counts = [
            [(0, 1, 1), (1, 0, 0), (0, 1, 1)], 
            [(1, 0, 0), (0, 1, 1), (0, 1, 1)],
        ]
        self.check_group_assignment(
            count_fn, groups_b[::-1], groups_a,
            correct_idxs, correct_counts
        )

    def check_group_assignment(self, count_fn, groups_a, groups_b,
                               correct_idxs, correct_counts):
        assign_idxs, counts = calc_group_assignment(
            groups_a, groups_b, count_fn
        )
        self.assertEqual(assign_idxs, correct_idxs)
        self.assertEqual(counts, correct_counts)

    def test_create_joint_groups(self):
        joint_groups = create_joint_groups([(self.entities, self.base_groups)])[0]
        self.assertEqual(joint_groups, self.joint_base_groups)

    def test_metrics_correct(self):
        preds = [self.base_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, self.correct_expected, group_metrics)

    def test_metrics_incorrect(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 20,
                    "true_positives": 0,
                    "false_positives": 10,
                    "false_negatives": 20,
                },
                "class2": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 10,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 10,
                },
            },
            "avg_f1": {
                "micro": 0,
                "macro": 0,
                "weighted": 0,
            }
        }
        preds = [self.incorrect_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)
        
    def test_metrics_overpredict(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 0.66666,
                    "recall": 1,
                    "f1-score": 0.8,
                    "support": 20,
                    "true_positives": 20,
                    "false_positives": 10,
                    "false_negatives": 0,
                },
                "class2": {
                    "precision": 1,
                    "recall": 1,
                    "f1-score": 1,
                    "support": 10,
                    "true_positives": 10,
                    "false_positives": 0,
                    "false_negatives": 0,
                },
            },
            "avg_f1": {
                "micro": 0.85714,
                "macro": 0.9,
                "weighted": 0.86666,
            }
        }
        preds = [self.overpredict_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_metrics_underpredict(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 1,
                    "recall": 0.5,
                    "f1-score": 0.66666,
                    "support": 20,
                    "true_positives": 10,
                    "false_positives": 0,
                    "false_negatives": 10,
                },
                "class2": {
                    "precision": 1,
                    "recall": 1,
                    "f1-score": 1,
                    "support": 10,
                    "true_positives": 10,
                    "false_positives": 0,
                    "false_negatives": 0,
                },
            },
            "avg_f1": {
                "micro": 0.8,
                "macro": 0.83333,
                "weighted": 0.77777,
            }
        }
        preds = [self.underpredict_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_metrics_under_over_predict(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 1,
                    "recall": 0.5,
                    "f1-score": 0.66666,
                    "support": 20,
                    "true_positives": 10,
                    "false_positives": 0,
                    "false_negatives": 10,
                },
                "class2": {
                    "precision": 0.3333,
                    "recall": 1,
                    "f1-score": 0.5,
                    "support": 10,
                    "true_positives": 10,
                    "false_positives": 20,
                    "false_negatives": 0,
                },
            },
            "avg_f1": {
                "micro": 0.57143,
                "macro": 0.58333,
                "weighted": 0.61111,
            }
        }
        preds = [self.under_over_predict_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_metrics_incorrect_labels(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 20,
                    "true_positives": 0,
                    "false_positives": 10,
                    "false_negatives": 20,
                },
                "class2": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 10,
                    "true_positives": 0,
                    "false_positives": 20,
                    "false_negatives": 10,
                },
            },
            "avg_f1": {
                "micro": 0,
                "macro": 0,
                "weighted": 0,
            }
        }
        preds = [self.incorrect_label_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_metrics_new_label(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 20,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 20,
                },
                "class2": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 10,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 10,
                },
                "class3": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 10,
                    "true_positives": 0,
                    "false_positives": 30,
                    "false_negatives": 0,
                },
            },
            "avg_f1": {
                "micro": 0,
                "macro": 0,
                "weighted": 0,
            }
        }
        preds = [self.new_label_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_metrics_mixed(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 0.66666,
                    "recall": 0.5,
                    "f1-score": 0.57143,
                    "support": 20,
                    "true_positives": 10,
                    "false_positives": 5,
                    "false_negatives": 10,
                },
                "class2": {
                    "precision": 1,
                    "recall": 0.5,
                    "f1-score": 0.66666,
                    "support": 10,
                    "true_positives": 5,
                    "false_positives": 0,
                    "false_negatives": 5,
                },
            },
            "avg_f1": {
                "micro": 0.6,
                "macro": 0.61905,
                "weighted": 0.60317,
            }
        }
        preds = [self.base_groups] * 5 + [self.incorrect_groups] * 5
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_metrics_mixed_labels(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 0.66666,
                    "recall": 0.5,
                    "f1-score": 0.57143,
                    "support": 20,
                    "true_positives": 10,
                    "false_positives": 5,
                    "false_negatives": 10,
                },
                "class2": {
                    "precision": 0.33333,
                    "recall": 0.5,
                    "f1-score": 0.4,
                    "support": 10,
                    "true_positives": 5,
                    "false_positives": 10,
                    "false_negatives": 5,
                },
            },
            "avg_f1": {
                "micro": 0.5,
                "macro": 0.48571,
                "weighted": 0.51429,
            }
        }
        preds = [self.base_groups] * 5 + [self.incorrect_label_groups] * 5
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_metrics_mixed_new_labels(self):
        expected = {
            "per_class_metrics": {
                "class1": {
                    "precision": 1,
                    "recall": 0.5,
                    "f1-score": 0.66666,
                    "support": 20,
                    "true_positives": 10,
                    "false_positives": 0,
                    "false_negatives": 10,
                },
                "class2": {
                    "precision": 1,
                    "recall": 0.5,
                    "f1-score": 0.66666,
                    "support": 10,
                    "true_positives": 5,
                    "false_positives": 0,
                    "false_negatives": 5,
                },
                "class3": {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 0,
                    "true_positives": 0,
                    "false_positives": 15,
                    "false_negatives": 0,
                },
            },
            "avg_f1": {
                "micro": 0.5,
                "macro": 0.44444,
                "weighted": 0.66666,
            }
        }
        preds = [self.base_groups] * 5 + [self.new_label_groups] * 5
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, expected, get_metrics)

    def test_group_metrics(self):
        # Simple integration test, as group_metrics is mostly an alias for
        # get_metrics, which is tested seperately
        preds = [self.base_groups] * 10
        labels = [self.base_groups] * 10
        self.check_metrics(preds, labels, self.correct_expected, group_metrics)

    def test_joint_metrics(self):
        # Simple integration test, as joint_metrics is mostly an alias for
        # get_metrics and create_joint_groups which are tested seperately
        preds = [(self.entities, self.base_groups)] * 10
        labels = [(self.entities, self.base_groups)] * 10
        self.check_metrics(preds, labels, self.correct_expected, joint_metrics)

    def check_metrics(self, preds, labels, expected, metric_fn):
        # Check per-class metrics
        per_class_metrics = metric_fn(preds, labels)
        for cls, correct_metrics in expected["per_class_metrics"].items():
            metrics = per_class_metrics[cls]
            for metric in ("precision", "recall", "f1-score"):
                self.assertAlmostEqual(
                    metrics[metric], correct_metrics[metric], places=3,
                    msg=f"{metric} in {cls} is incorrect!"
                )
            for metric in ("true_positives", "false_positives", "false_negatives"):
                self.assertEqual(
                    metrics[metric], correct_metrics[metric],
                    msg=f"{metric} in {cls} is incorrect!"
                )

        # Check average metrics
        for avg, correct_f1 in expected["avg_f1"].items():
            avg_f1 = metric_fn(preds, labels, average=avg)
            self.assertAlmostEqual(
                avg_f1, correct_f1, places=3,
                msg=f"{avg} f1 in is incorrect!"
            )


class TestGroupMetricCountFunctions(unittest.TestCase):
    def setUp(self):
        # Each line is 7 tokens, (x, percent, (, i, %, ), \n)
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        group_a = {
            "spans": [
                {"start": 0, "end": 39, "text": "five percent (5%) \n fifty percent (50%)"},
                {"start": 61, "end": 78, "text": "nine percent (9%)"},
            ],
            "label": "class1"
        }
        group_b = {
            "spans": [
                {"start": 42, "end": 58, "text": "two percent (2%)"},
                {"start": 81, "end": 99, "text": "three percent (3%)"},
            ],
            "label": "class1"
        }
        group_overlap = {
            "spans": [
                {"start": 20, "end": 58, "text": "fifty percent (50%) \n two percent (2%)"},
                # Non overlapping span for joint metric tests
                {"start": 42, "end": 58, "text": "two percent (2%)"},
                {"start": 66, "end": 99, "text": "percent (9%) \n three percent (3%)"},
            ],
            "label": "class1"
        }
        group_superset = {
            "spans": [
                {"start": 0, "end": 53, "text": text[0:53]},
                # Non superset span for joint metric tests
                {"start": 42, "end": 58, "text": "two percent (2%)"},
                {"start": 46, "end": 86, "text": text[46:86]},
            ],
            "label": "class1"
        }
        group_tokens = {
            "spans": [
                {"start": 0, "end": 4, "text": "five"},
                {"start": 54, "end": 58, "text": "(3%)"},
                {"start": 74, "end": 78, "text": "(9%)"},
            ],
            "label": "class1"
        }
        group_all = {
            "spans": [
                {"start": 0, "end": 102, "text": text},
            ],
            "label": "class1"
        }
        # Associate groups with keys so we can do set operations
        self.groups = {
            "group_a": group_a,
            "group_b": group_b,
            "group_overlap": group_overlap,
            "group_superset": group_superset,
            "group_tokens": group_tokens,
            "group_all": group_all,
        }

        # Make versions of all groups with no group label
        # Used to check label mismatch behavior
        no_label_groups = {}
        for name, group in self.groups.items():
            group = deepcopy(group)
            group["label"] = None
            no_label_groups[name + "_nl"] = group
        self.groups.update(no_label_groups)

        # Make versions with entities by assigning different labels
        # Used to check joint metrics
        self.joint_groups = {}
        # Labels to be assigned to spans in groups
        correct_labels = ["a", "a", "a", "a"]
        incorrect_labels = ["b", "b", "b", "b"]
        mix_labels = ["a", "b", "a", "b"]
        mix_inv_labels = mix_labels[::-1]
        for name, group in self.groups.items():
            name = name.replace("group", "joint")
            num_spans = len(group["spans"])
            # All correct labels
            self.joint_groups[name + "_cl"] = self.add_entities(
                group, correct_labels[:num_spans]
            )
            # All wrong labels
            self.joint_groups[name + "_icl"] = self.add_entities(
                group, incorrect_labels[:num_spans]
            )
            # Mixed labels, doesn't apply to groups with only 1 span
            if num_spans >= 2:
                self.joint_groups[name + "_ml"] = self.add_entities(
                    group, mix_labels[:num_spans]
                )
                self.joint_groups[name + "_mil"] = self.add_entities(
                    group, mix_inv_labels[:num_spans]
                )

    def add_entities(self, group, labels):
        # Make a copy with an "entities" field added
        group = deepcopy(group)
        spans = deepcopy(group["spans"])
        group["entities"] = spans
        if len(spans) != len(labels):
            raise ValueError(
                "Number of labels and number of spans must match!"
            )
        for span, label in zip(spans, labels):
            span["label"] = label
        return group

    def test_group_exact_counts(self):
        match_groups = [
            "group_a"
        ]
        self.check_group_counts("group_a", "exact", match_groups)

    def test_group_overlap_counts(self):
        match_groups = [
            "group_a",
            "group_overlap",
            "group_superset",
            "group_tokens",
            "group_all",
        ]
        self.check_group_counts("group_a", "overlap", match_groups)
    
    def test_group_superset_counts(self):
        match_groups = [
            "group_a",
            "group_superset",
            "group_all",
        ]
        # Superset matches don't go both ways, so we turn check_swap off
        self.check_group_counts("group_a", "superset", match_groups,
                                check_swap=False)

    def test_group_token_counts(self):
        # Count tuples are of the form (TP, FP, FN)
        # All pred tokens are false positives and all ground truth tokens are
        # false negatives when labels are mismatched
        
        # The group to compare against
        base_group = "group_a"

        self.check_token_counts("group_a", base_group, (19, 0, 0))
        self.check_token_counts("group_a_nl", base_group, (0, 19, 19))

        self.check_token_counts("group_b", base_group, (0, 12, 19))
        self.check_token_counts("group_b_nl", base_group, (0, 12, 19))

        # Note", the tokens "two percent" appear in several text spans of the
        # overlap and superset groups, and are therefore counted as false
        # positives multiple times

        self.check_token_counts("group_overlap", base_group, (11, 20, 8))
        self.check_token_counts("group_overlap_nl", base_group, (0, 31, 19))

        self.check_token_counts("group_superset", base_group, (19, 17, 0))
        self.check_token_counts("group_superset_nl", base_group, (0, 36, 19))

        self.check_token_counts("group_tokens", base_group, (5, 4, 14))
        self.check_token_counts("group_tokens_nl", base_group, (0, 9, 19))

        self.check_token_counts("group_all", base_group, (19, 16, 0))
        self.check_token_counts("group_all_nl", base_group, (0, 35, 19))

    def test_joint_exact_counts(self):
        match_groups = [
            "joint_a_cl"
        ]
        self.check_group_counts("joint_a_cl", "exact", match_groups,
                                metric_type="joint")
        
    def test_joint_overlap_counts(self):
        match_groups = [
            "joint_a_cl", "joint_a_ml", "joint_a_mil",
            "joint_overlap_cl", "joint_overlap_ml",
            "joint_superset_cl", "joint_superset_ml",
            "joint_tokens_cl", "joint_tokens_ml",
            "joint_all_cl",
        ]
        self.check_group_counts("joint_a_cl", "overlap", match_groups,
                                metric_type="joint")

    def test_joint_supset_counts(self):
        match_groups = [
            "joint_a_cl",
            "joint_superset_cl", "joint_superset_ml",
            "joint_all_cl",
        ]
        self.check_group_counts("joint_a_cl", "superset", match_groups,
                                metric_type="joint", check_swap=False)

    def test_joint_token_counts(self):
        # Count tuples are of the form (TP, FP, FN)
        # All pred tokens are false positives and all ground truth tokens are
        # false negatives when labels are mismatched

        # Avoid calling metric_type every time
        check_token_counts = lambda x, y, z: self.check_token_counts(
            x, y, z, metric_type="joint"
        )

        # The group to compare against
        base_group = "joint_a_cl"

        
        check_token_counts("joint_a_cl", base_group, (19, 0, 0))
        check_token_counts("joint_a_icl", base_group, (0, 19, 19))
        check_token_counts("joint_a_ml", base_group, (13, 6, 6))
        check_token_counts("joint_a_mil", base_group, (6, 13, 13))
        # Always the same counts when group labels are mismatched
        for tl in ["cl", "icl", "ml", "mil"]:
            check_token_counts(
                "joint_a_nl_" + tl, base_group, (0, 19, 19)
            )

        # Group b doesn't overlap with the base group
        for gl in ["", "nl_"]:
            for tl in ["cl", "icl", "ml", "mil"]:
                check_token_counts(
                    "joint_b_" + gl + tl, base_group, (0, 12, 19)
                )

        # Note", the tokens "two percent" appear in several text spans of the
        # overlap and superset joints, and are therefore counted as false
        # positives multiple times

        check_token_counts("joint_overlap_cl", base_group, (11, 20, 8))
        check_token_counts("joint_overlap_icl", base_group, (0, 31, 19))
        check_token_counts("joint_overlap_ml", base_group, (11, 20, 8))
        check_token_counts("joint_overlap_mil", base_group, (0, 31, 19))
        for tl in ["cl", "icl", "ml", "mil"]:
            check_token_counts(
                "joint_overlap_nl_" + tl, base_group, (0, 31, 19)
            )

        check_token_counts("joint_superset_cl", base_group, (19, 17, 0))
        check_token_counts("joint_superset_icl", base_group, (0, 36, 19))
        check_token_counts("joint_superset_ml", base_group, (19, 17, 0))
        check_token_counts("joint_superset_mil", base_group, (0, 36, 19))
        for tl in ["cl", "icl", "ml", "mil"]:
            check_token_counts(
                "joint_superset_nl_" + tl, base_group, (0, 36, 19)
            )

        check_token_counts("joint_tokens_cl", base_group, (5, 4, 14))
        check_token_counts("joint_tokens_icl", base_group, (0, 9, 19))
        check_token_counts("joint_tokens_ml", base_group, (5, 4, 14))
        check_token_counts("joint_tokens_mil", base_group, (0, 9, 19))
        for tl in ["cl", "icl", "ml", "mil"]:
            check_token_counts(
                "joint_tokens_nl_" + tl, base_group, (0, 9, 19)
            )

        check_token_counts("joint_all_cl", base_group, (19, 16, 0))
        check_token_counts("joint_all_icl", base_group, (0, 35, 19))
        for tl in ["cl", "icl"]:
            check_token_counts(
                "joint_all_nl_" + tl, base_group, (0, 35, 19)
            )

    def check_group_counts(self, group, span_type, match_groups,
                           check_swap=True, metric_type="group"):
        """
        :param group: String, the ground truth group to check against
        :param span: String, the type of span to check
        :param match_groups: List of strings, the set of groups that should
        match the ground truth group under the current count_fn
        :param check_swap: Bool, whether or not the inverse (ground truth as
        pred) should also be checked
        :param metric_type: String, whether we should convert the groups using
        the groups dictionary or joint groups dicionary
        """
        count_fn = get_count_fn(metric_type, span_type)

        # Get dictionary to convert from str -> group
        if metric_type == "group":
            all_groups = self.groups
        elif metric_type == "joint":
            all_groups = self.joint_groups

        # Get groups that shouldn't match
        dif_groups = set(all_groups.keys()) - set(match_groups)

        # Strings to groups
        group = all_groups[group]
        match_groups = [all_groups[n] for n in match_groups]
        dif_groups = [all_groups[n] for n in dif_groups]

        for match_group in match_groups:
            # Correct matches should return 1 TP, 0 FP, 0 FN
            self.check_counts(count_fn, match_group, group, (1, 0, 0),
                              check_swap=check_swap)
        for dif_group in dif_groups:
            # Incorrect matches should return 0 TP, 1 FP, 1 FN
            self.check_counts(count_fn, dif_group, group, (0, 1, 1),
                              check_swap=check_swap)

    def check_token_counts(self, pred_group, label_group, correct_counts,
                           metric_type="group"):
        """
        :param pred_group, label_group: Strings, the groups to check
        :param correct_counts: Tuple of (TP, FP, FN), the expected counts
        :param metric_type: String, whether we should convert the groups using
        the groups dictionary or joint groups dicionary
        """
        count_fn = get_count_fn(metric_type, "token")

        # Get dictionary to convert from str -> group
        if metric_type == "group":
            all_groups = self.groups
        elif metric_type == "joint":
            all_groups = self.joint_groups

        pred_group = all_groups[pred_group]
        label_group = all_groups[label_group]

        self.check_counts(count_fn, pred_group, label_group, correct_counts)

    def check_counts(self, count_fn, pred_group, label_group, correct_counts,
                     check_swap=True):
        counts = count_fn(pred_group, label_group)
        self.assertEqual(
            counts, correct_counts,
            f"\n{pred_group}\nand\n{label_group}\nmatch incorrectly!"
        )

        if check_swap:
            # Ensure consistent behavior with pred / label swapped
            # Swap false positives and false negatives
            inv_correct_counts = (
                correct_counts[0], correct_counts[2], correct_counts[1]
            )
            inv_counts = count_fn(label_group, pred_group)
            self.assertEqual(
                inv_counts, inv_correct_counts,
                (f"\n{pred_group}\nand\n{label_group}\ncounts are not consistent "
                 f"when swapping pred / label!")
            )

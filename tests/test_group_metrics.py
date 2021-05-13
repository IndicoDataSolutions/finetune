import unittest
from copy import deepcopy
from finetune.util.group_metrics import (
    get_count_fn
)

class TestGroupMetrics(unittest.TestCase):
    def setUp(self):
        pass

    def test_group_assignment(self):
        pass

    def test_create_joint_groups(self):
        pass

    def test_group_metrics(self):
        pass

    def test_joint_metrics(self):
        pass

class TestGroupMetricCountFunctions(unittest.TestCase):
    def setUp(self):
        # Each line is 7 tokens, (x, percent, (, i, %, ), \n)
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        group_a = {
            "tokens": [
                {"start": 0, "end": 39, "text": "five percent (5%) \n fifty percent (50%)"},
                {"start": 61, "end": 78, "text": "nine percent (9%)"},
            ],
            "label": "class1"
        }
        group_b = {
            "tokens": [
                {"start": 42, "end": 58, "text": "two percent (2%)"},
                {"start": 81, "end": 99, "text": "three percent (3%)"},
            ],
            "label": "class1"
        }
        group_overlap = {
            "tokens": [
                {"start": 20, "end": 58, "text": "fifty percent (50%) \n two percent (2%)"},
                # Non overlapping span for joint metric tests
                {"start": 42, "end": 58, "text": "two percent (2%)"},
                {"start": 66, "end": 99, "text": "percent (9%) \n three percent (3%)"},
            ],
            "label": "class1"
        }
        group_superset = {
            "tokens": [
                {"start": 0, "end": 53, "text": text[0:53]},
                # Non overlapping span for joint metric tests
                {"start": 42, "end": 58, "text": "two percent (2%)"},
                {"start": 46, "end": 86, "text": text[46:86]},
            ],
            "label": "class1"
        }
        group_tokens = {
            "tokens": [
                {"start": 0, "end": 4, "text": "five"},
                {"start": 54, "end": 58, "text": "(3%)"},
                {"start": 74, "end": 78, "text": "(9%)"},
            ],
            "label": "class1"
        }
        group_all = {
            "tokens": [
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
            no_label_groups[name + "_nl"] = self.no_label(group)
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
            num_spans = len(group["tokens"])
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

    def no_label(self, group):
        # Make a copy without the label
        group = deepcopy(group)
        group["label"] = None
        return group

    def add_entities(self, group, labels):
        # Make a copy with an "entities" field added
        group = deepcopy(group)
        spans = deepcopy(group["tokens"])
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




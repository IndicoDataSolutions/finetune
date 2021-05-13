import unittest
from copy import deepcopy
from finetune.util.group_metrics import (
    get_count_fn
)

class TestGroupMetrics(unittest.TestCase):
    def setUp(self):
        # Each line is 6 tokens, (x, percent, (, i, %, )), newline is 1 token
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        self.group_a = {
            "tokens": [
                {"start": 0, "end": 39, "text": "five percent (5%) \n fifty percent (50%)"},
                {"start": 61, "end": 78, "text": "nine percent (9%)"},
            ],
            "label": "class1"
        }
        self.group_b = {
            "tokens": [
                {"start": 42, "end": 58, "text": "two percent (2%)"},
                {"start": 81, "end": 99, "text": "three percent (3%)"},
            ],
            "label": "class1"
        }
        self.group_overlap = {
            "tokens": [
                {"start": 20, "end": 58, "text": "fifty percent (50%) \n two percent (2%)"},
                {"start": 66, "end": 99, "text": "percent (9%) \n three percent (3%)"},
            ],
            "label": "class1"
        }
        self.group_superset = {
            "tokens": [
                {"start": 0, "end": 53, "text": text[0:53]},
                {"start": 42, "end": 86, "text": text[42:83]},
            ],
            "label": "class1"
        }
        self.group_tokens = {
            "tokens": [
                {"start": 0, "end": 4, "text": "five"},
                {"start": 26, "end": 33, "text": "percent"},
                {"start": 74, "end": 78, "text": "(9%)"},
                {"start": 95, "end": 99, "text": "(3%)"},
            ],
            "label": "class1"
        }
        self.group_all = {
            "tokens": [
                {"start": 0, "end": 102, "text": text},
            ],
            "label": "class1"
        }

        self.group_a_nl = self.no_label(self.group_a)
        self.group_b_nl = self.no_label(self.group_b)
        self.group_overlap_nl = self.no_label(self.group_overlap)
        self.group_superset_nl = self.no_label(self.group_superset)
        self.group_tokens_nl = self.no_label(self.group_tokens)
        self.group_all_nl = self.no_label(self.group_all)

    def no_label(self, group):
        group = deepcopy(group)
        group["label"] = None
        return group

    def test_group_exact_counts(self):
        count_fn = get_count_fn("group", "exact")
        match_groups = [
            self.group_a
        ]
        dif_groups = [
            self.group_a_nl,
            self.group_b, self.group_b_nl,
            self.group_overlap, self.group_overlap_nl,
            self.group_superset, self.group_superset_nl,
            self.group_tokens, self.group_tokens_nl,
            self.group_all, self.group_all_nl,
        ]
        self.check_group_counts(
            self.group_a, count_fn, match_groups, dif_groups
        )

    def test_group_overlap_group_counts(self):
        count_fn = get_count_fn("group", "overlap")
        match_groups = [
            self.group_a,
            self.group_overlap,
            self.group_superset,
            self.group_tokens,
            self.group_all,
        ]
        dif_groups = [
            self.group_a_nl,
            self.group_b, self.group_b_nl,
            self.group_overlap_nl,
            self.group_superset_nl,
            self.group_tokens_nl,
            self.group_all_nl,
        ]
        self.check_group_counts(
            self.group_a, count_fn, match_groups, dif_groups
        )
    
    def test_group_superset_group_counts(self):
        count_fn = get_count_fn("group", "superset")
        match_groups = [
            self.group_a,
            self.group_superset,
            self.group_all,
        ]
        dif_groups = [
            self.group_a_nl,
            self.group_b, self.group_b_nl,
            self.group_overlap, self.group_overlap_nl,
            self.group_superset_nl,
            self.group_tokens, self.group_tokens_nl,
            self.group_all_nl,
        ]
        # Superset matches don't go both ways, so we turn check_swap off
        self.check_group_counts(
            self.group_a, count_fn, match_groups, dif_groups, check_swap=False,
        )

    def test_group_token_counts(self):
        # Count tuples are of the form (TP, FP, FN)
        # All pred tokens are false positives and all ground truth tokens are
        # false negatives when labels are mismatched
        self.check_token_counts(self.group_a, self.group_a, (19, 0, 0))
        self.check_token_counts(self.group_a_nl, self.group_a, (0, 19, 19))

        self.check_token_counts(self.group_b, self.group_a, (0, 12, 19))
        self.check_token_counts(self.group_b_nl, self.group_a, (0, 12, 19))

        self.check_token_counts(self.group_overlap, self.group_a, (11, 14, 8))
        self.check_token_counts(self.group_overlap_nl, self.group_a, (0, 25, 19))

        # Note, the tokens "two percent" appear in both text spans of the
        # superset group, and are therefore counted as false positives twice
        self.check_token_counts(self.group_superset, self.group_a, (19, 12, 0))
        self.check_token_counts(self.group_superset_nl, self.group_a, (0, 31, 19))

        self.check_token_counts(self.group_tokens, self.group_a, (6, 4, 13))
        self.check_token_counts(self.group_tokens_nl, self.group_a, (0, 10, 19))

        self.check_token_counts(self.group_all, self.group_a, (19, 16, 0))
        self.check_token_counts(self.group_all_nl, self.group_a, (0, 35, 19))

    def check_group_counts(self, group, count_fn, match_groups, dif_groups,
                           check_swap=True):
        for match_group in match_groups:
            # Correct matches should return 1 TP, 0 FP, 0 FN
            self.check_counts(count_fn, match_group, group, (1, 0, 0),
                              check_swap=check_swap)
        for dif_group in dif_groups:
            # Incorrect matches should return 0 TP, 1 FP, 1 FN
            self.check_counts(count_fn, dif_group, group, (0, 1, 1),
                              check_swap=check_swap)

    def check_token_counts(self, pred_group, label_group, correct_counts):
        count_fn = get_count_fn("group", "token")
        self.check_counts(count_fn, pred_group, label_group, correct_counts)

    def check_counts(self, count_fn, pred_group, label_group, correct_counts,
                     check_swap=True):
        counts = count_fn(pred_group, label_group)
        self.assertEqual(
            counts, correct_counts,
            f"\n{pred_group} \nand \n{label_group} \nmatch incorrectly!"
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




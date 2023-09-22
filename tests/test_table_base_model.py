import chunk
import os
import unittest

from finetune import SequenceLabeler
from finetune.base_models import TableRoBERTa
from finetune.base_models.bert.table_utils import (
    batch_packing,
    get_gather_indices,
    slice_by_table_indices,
    gather_col_vals,
    scatter_feats,
    get_summary_values,
)

import tensorflow as tf

DATA_PATH = os.path.join("tests", "data", "doc_rep_integration.csv")


class TestTableModel(unittest.TestCase):
    def setUp(self) -> None:
        self.text = [
            "Item Number Description\n"
            + "1 Chair\n"
            + "2 Table\n"
            + "3 Monitor\n"
            + "Spanning Multiple Cells"
        ]
        self.labels = [
            [
                {"start": 24, "end": 25, "label": "item number", "text": "1"},
                {"start": 32, "end": 33, "label": "item number", "text": "2"},
                {"start": 40, "end": 41, "label": "item number", "text": "3"},
            ]
        ]
        self.context = [
            [
                {
                    "start": 0,
                    "end": 11,
                    "start_row": 0,
                    "end_row": 0,
                    "start_col": 0,
                    "end_col": 0,
                    "text": "Item Number",
                },
                {
                    "start": 12,
                    "end": 23,
                    "start_row": 0,
                    "end_row": 0,
                    "start_col": 1,
                    "end_col": 1,
                    "text": "Description",
                },
                {
                    "start": 24,
                    "end": 25,
                    "start_row": 1,
                    "end_row": 1,
                    "start_col": 0,
                    "end_col": 0,
                    "text": "1",
                },
                {
                    "start": 26,
                    "end": 31,
                    "start_row": 1,
                    "end_row": 1,
                    "start_col": 1,
                    "end_col": 1,
                    "text": "Chair",
                },
                {
                    "start": 32,
                    "end": 33,
                    "start_row": 2,
                    "end_row": 2,
                    "start_col": 0,
                    "end_col": 0,
                    "text": "2",
                },
                {
                    "start": 34,
                    "end": 39,
                    "start_row": 2,
                    "end_row": 2,
                    "start_col": 1,
                    "end_col": 1,
                    "text": "Table",
                },
                {
                    "start": 40,
                    "end": 41,
                    "start_row": 3,
                    "end_row": 3,
                    "start_col": 0,
                    "end_col": 0,
                    "text": "3",
                },
                {
                    "start": 42,
                    "end": 49,
                    "start_row": 3,
                    "end_row": 3,
                    "start_col": 1,
                    "end_col": 1,
                    "text": "Monitor",
                },
                {
                    "start": 50,
                    "end": 73,
                    "start_row": 4,
                    "end_row": 4,
                    "start_col": 0,
                    "end_col": 1,
                    "text": "Spanning Multiple Cells",
                },
            ]
        ]

    def test_fit_predict(self):
        model = SequenceLabeler(base_model=TableRoBERTa)
        model.fit(self.text * 20, self.labels * 20, context=self.context * 20)
        preds = model.predict(self.text, context=self.context)[0]
        assert len(preds) == 3
        assert set(p["text"] for p in preds) == set(l["text"] for l in self.labels[0])


class TestTableUtils:
    def test_batch_packing(self):
        output_ragged, mask, pos_ids = batch_packing(
            tf.ragged.constant(
                [
                    [1, 2],
                    [3],
                    [11],
                    [14],
                    [4, 5, 6, 7],
                    [101, 102, 103, 104, 105, 106, 107, 108, 109],
                ]
            )
        )

        assert tf.reduce_all(
            tf.equal(
                output_ragged,
                tf.ragged.constant(
                    [
                        [1, 2, 3, 11, 14, 4, 5, 6, 7],
                        [101, 102, 103, 104, 105, 106, 107, 108, 109],
                    ]
                ),
            )
        )
        assert mask.shape == (2, 9, 9)
        assert tf.reduce_all(
            mask
            == tf.constant(
                [
                    [
                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                ]
            )
        )
        assert tf.reduce_all(
            pos_ids.to_tensor()
            == tf.constant(
                [[0, 1, 0, 0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
                dtype=tf.int64,
            )
        )

    def test_get_gather_indices(self):
        gi = get_gather_indices(
            X=tf.constant([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, -1]]),
            sequence_lengths=tf.constant([7, 6]),
            start=tf.constant([[0, 1, 0, 1, 0, 1, 0], [0, 1, 2, 0, 1, 2, -1]]),
            end=tf.constant([[0, 2, 0, 2, 0, 2, 0], [0, 1, 2, 0, 1, 2, -1]]),
            other_end=tf.constant([[0, 2, 0, 2, 0, 2, 0], [0, 1, 2, 0, 1, 2, -1]]),
            chunk_tables=False,
        )
        assert tf.reduce_all(
            gi["seq_lens"] == tf.constant([6, 4, 5, 4, 5, 4], dtype=tf.int64)
        )
        assert tf.reduce_all(
            gi["values"]
            == tf.constant(
                [
                    [[0, 8], [0, 0], [0, 2], [0, 4], [0, 6], [0, 7]],
                    [[0, 8], [1, 0], [1, 3], [0, 7], [0, 9], [0, 9]],
                    [[0, 8], [0, 1], [0, 3], [0, 5], [0, 7], [0, 9]],
                    [[0, 8], [1, 1], [1, 4], [0, 7], [0, 9], [0, 9]],
                    [[0, 8], [0, 1], [0, 3], [0, 5], [0, 7], [0, 9]],
                    [[0, 8], [1, 2], [1, 5], [0, 7], [0, 9], [0, 9]],
                ],
                dtype=gi["values"].dtype,
            )
        )
        # attn mask and pos IDs come directly from batch packing which is tested elsewhere.

    def test_get_summary_values(self):
        summary_vals = get_summary_values(
            # Batch packing would never do this, but just to keep the test easy
            inp=[
                [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35],
                [40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55],
            ],
            gather_vals=[
                [
                    [0, 8],
                    [0, 0],
                    [0, 2],
                    [0, 4],
                    [0, 6],
                    [0, 7],
                    [0, 8],
                    [1, 0],
                    [1, 3],
                    [0, 7],
                    [0, 9],
                    [0, 9],
                ],
                [
                    [0, 8],
                    [0, 1],
                    [0, 3],
                    [0, 5],
                    [0, 7],
                    [0, 9],
                    [0, 8],
                    [1, 1],
                    [1, 4],
                    [0, 7],
                    [0, 9],
                    [0, 9],
                ],
                [
                    [0, 8],
                    [0, 1],
                    [0, 3],
                    [0, 5],
                    [0, 7],
                    [0, 9],
                    [0, 8],
                    [1, 2],
                    [1, 5],
                    [0, 7],
                    [0, 9],
                    [0, 9],
                ],
            ],
            input_seq_len=7,
        )
        assert tf.reduce_all(
            summary_vals
            == tf.constant(
                [
                    [0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10],
                    [20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30],
                    [40, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50],
                ],
                dtype=summary_vals.dtype,
            )
        )

    def test_slice_by_table_indices(self):
        gi = slice_by_table_indices(
            tf.constant(
                [
                    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
                    [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]],
                ]
            ),
            tf.constant(
                [
                    [
                        [True, False, True, False, True, False, True],
                        [False, True, False, True, False, True, False],
                    ],
                    [
                        [True, True, True, True, True, True, True],
                        [False, False, False, False, False, False, False],
                    ],
                ],
                dtype=tf.bool,
            ),
            eos_pad=tf.convert_to_tensor([0, 99]),
            bos_pad=tf.convert_to_tensor([0, 100]),
            pad_val=tf.convert_to_tensor([0, 101]),
            other_end=tf.constant([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]),
            include_mask=True,
        )
        assert tf.reduce_all(
            gi["values"]
            == tf.constant(
                [
                    [
                        [0, 100],
                        [0, 0],
                        [0, 2],
                        [0, 4],
                        [0, 6],
                        [0, 99],
                        [0, 101],
                        [0, 101],
                        [0, 101],
                    ],
                    [
                        [0, 100],
                        [1, 1],
                        [1, 3],
                        [1, 5],
                        [0, 99],
                        [0, 101],
                        [0, 101],
                        [0, 101],
                        [0, 101],
                    ],
                    [
                        [0, 100],
                        [0, 0],
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [0, 4],
                        [0, 5],
                        [0, 6],
                        [0, 99],
                    ],
                ],
                dtype=gi["values"].dtype,
            )
        )

    def test_slice_by_table_indices_chunking(self):
        gi = slice_by_table_indices(
            tf.constant([[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]]]),
            tf.constant(
                [[[True, True, True, False, True, False, True]]], dtype=tf.bool
            ),
            eos_pad=tf.convert_to_tensor([0, 99]),
            bos_pad=tf.convert_to_tensor([0, 100]),
            pad_val=tf.convert_to_tensor([0, 101]),
            include_mask=True,
            other_end=tf.constant([[0, 1, 2, 3, 4, 5, 6]]),
            max_length=5,
        )
        print(gi["values"].numpy())
        assert tf.reduce_all(
            gi["values"]
            == tf.constant(
                [
                    [[0, 100], [0, 0], [0, 1], [0, 2], [0, 99]],
                    [[0, 100], [0, 0], [0, 1], [0, 4], [0, 99]],
                    [[0, 100], [0, 0], [0, 1], [0, 6], [0, 99]],
                ],
                dtype=gi["values"].dtype,
            )
        )

    def test_slice_by_table_indices_chunking_length_fallback(self):
        gi = slice_by_table_indices(
            tf.constant([[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]]]),
            tf.constant([[[True, True, True, True, True, True, True]]], dtype=tf.bool),
            eos_pad=tf.convert_to_tensor([0, 99]),
            bos_pad=tf.convert_to_tensor([0, 100]),
            pad_val=tf.convert_to_tensor([0, 101]),
            include_mask=True,
            other_end=tf.constant([[0, 0, 1, 1, 4, 5, 6]]),
            max_length=5,
        )
        print(gi["values"].numpy())
        assert tf.reduce_all(
            gi["values"]
            == tf.constant(
                [
                    [[0, 100], [0, 0], [0, 1], [0, 2], [0, 99]],
                    [[0, 100], [0, 3], [0, 4], [0, 5], [0, 99]],
                    [[0, 100], [0, 6], [0, 99], [0, 101], [0, 101]],
                ],
                dtype=gi["values"].dtype,
            )
        )

    def test_gather_col_vals(self):
        output = gather_col_vals(
            inp=tf.constant([[0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15, 16]]),
            gather_output={
                "values": tf.constant(
                    [
                        [
                            [0, 7],
                            [0, 0],
                            [0, 2],
                            [0, 4],
                            [0, 6],
                            [0, 8],
                            [0, 9],
                            [0, 9],
                            [0, 9],
                        ],
                        [
                            [0, 7],
                            [1, 1],
                            [1, 3],
                            [1, 5],
                            [0, 8],
                            [0, 9],
                            [0, 9],
                            [0, 9],
                            [0, 9],
                        ],
                        [
                            [0, 7],
                            [0, 0],
                            [0, 1],
                            [0, 2],
                            [0, 3],
                            [0, 4],
                            [0, 5],
                            [0, 6],
                            [0, 8],
                        ],
                    ]
                ),
                "seq_lens": "seq_lens_dummy_val",
                "attn_mask": "attn_mask_dummy_val",
            },
            eos_pad=tf.constant(1000),
            bos_pad=tf.constant(1001),
            pad_val=tf.constant(-1000),
        )
        assert output["seq_lens"] == "seq_lens_dummy_val"
        assert output["attn_mask"] == "attn_mask_dummy_val"
        assert tf.reduce_all(
            output["values"]
            == tf.constant(
                [
                    [1001, 0, 2, 4, 6, 1000, -1000, -1000, -1000],
                    [1001, 11, 13, 15, 1000, -1000, -1000, -1000, -1000],
                    [1001, 0, 1, 2, 3, 4, 5, 6, 1000],
                ],
                dtype=output["values"].dtype,
            )
        )

    def test_scatter_feats(self):
        res = scatter_feats(
            output_shape=tf.constant([1, 3, 1]),
            sequence_feats=tf.constant(
                [[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]
            ),
            scatter_vals=tf.constant(
                [
                    [[0, 4], [0, 1], [0, 2], [0, 5]],  # bos, 1, 2, eos
                    [[0, 4], [0, 2], [0, 5], [0, 6]],  # bos, 2, eos, pad.
                ]
            ),
        )
        assert tf.reduce_all(
            res == tf.constant([[[0.0], [1.0], [(5 + 2) / 2]]], dtype=res.dtype)
        )

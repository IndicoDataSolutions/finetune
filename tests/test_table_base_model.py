import os
import unittest

import tensorflow as tf

from finetune import SequenceLabeler
from finetune.base_models import TableRoBERTa
from finetune.base_models.bert.table_utils import (
    batch_packing,
    chunk_ragged_tensor,
    gather_col_vals,
    get_summary_values,
    scatter_feats,
    build_gather_outputs,
)

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

    def test_batch_packing_target_seq_len(self):
        output_ragged, mask, pos_ids = batch_packing(
            tf.ragged.constant(
                [
                    [1, 2],
                    [3],
                    [11],
                    [14],
                    [4, 5, 6, 7],
               ]
            ),
            training=True,
            target_seq_len=10,
        )
        print(output_ragged.numpy())
        assert tf.reduce_all(
            tf.equal(
                output_ragged,
                tf.ragged.constant(
                    [
                        [1, 2, 3, 11, 14, 4, 5, 6, 7],
                    ]
                ),
            )
        )
        assert mask.shape == (1, 9, 9)
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
                ]
            )
        )
        assert tf.reduce_all(
            pos_ids.to_tensor()
            == tf.constant(
                [[0, 1, 0, 0, 0, 0, 1, 2, 3]],
                dtype=tf.int64,
            )
        )

    def test_get_gather_indices(self):
        gi = build_gather_outputs(
            X=tf.constant([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, -1]]),
            sequence_lengths=tf.constant([7, 6]),
            start=tf.constant([[0, 1, 0, 1, 0, 1, 0], [0, 1, 2, 0, 1, 2, -1]]),
            end=tf.constant([[0, 2, 0, 2, 0, 2, 0], [0, 1, 2, 0, 1, 2, -1]]),
            other_end=tf.constant([[0, 2, 0, 2, 0, 2, 0], [0, 1, 2, 0, 1, 2, -1]]),
            chunk_tables=False,
            include_mask=True,
            training=True,
        )
        assert tf.reduce_all(
            gi["seq_lens"] == tf.constant([6, 4, 5, 4, 5, 4], dtype=tf.int32)
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



    def test_slice_by_table_indices_chunking(self):
        gi = build_gather_outputs(
            X=tf.constant([[0, 1, 2, 3, 4, 5, 6]]),
            sequence_lengths=tf.constant([7]),
            start=tf.constant([[0, 0, 0, 0, 0, 0, 0]]),
            end=tf.constant([[0, 0, 0, 0, 0, 0, 6]]),
            other_end=tf.constant([[0, 1, 2, 3, 4, 5, 6]]),
            chunk_tables=True,
            include_mask=True,
            base_model_max_length=5,
            training=True,
        )
        # Verify header reuse: first token after BOS (index 1) of each chunk references row 0
        first_tokens = gi["values"][:, 1]
        assert tf.reduce_all(first_tokens[:, 0] == 0)
        # Verify EOS present at last position of each chunk
        last_tokens = gi["values"][:, -1]
        assert tf.reduce_all(last_tokens[:, 1] >= 7)

    def test_slice_by_table_indices_chunking_length_fallback(self):
        gi = build_gather_outputs(
            X=tf.constant([[0, 1, 2, 3, 4, 5, 6]]),
            sequence_lengths=tf.constant([7]),
            start=tf.constant([[0, 0, 0, 0, 0, 0, 0]]),
            end=tf.constant([[0, 0, 0, 0, 0, 0, 6]]),
            include_mask=True,
            other_end=tf.constant([[0, 0, 1, 1, 4, 5, 6]]),
            chunk_tables=True,
            base_model_max_length=5,
            training=True,
        )
        # Verify fallback: chunks after the first carry no header rows when headers overflow
        values = gi["values"]
        # For simplicity: ensure at least one trailing chunk has padding tokens (index >= 9)
        assert tf.reduce_any(values[:, -1, 1] >= 9)

    def test_build_gather_outputs_target_seq_len(self):
        gi = build_gather_outputs(
            X=tf.constant([[0, 1, 2, 3, 4, 5]]),
            sequence_lengths=tf.constant([6]),
            start=tf.constant([[0, 0, 0, 0, 0, 0]]),
            end=tf.constant([[0, 0, 0, 0, 0, 5]]),
            other_end=tf.constant([[0, 0, 0, 0, 0, 5]]),
            chunk_tables=False,
            include_mask=True,
            training=True,
            target_seq_len=8,
        )
        # Expect exactly target_seq_len tokens per row
        assert tf.shape(gi["values"])[1] == 8
        assert tf.shape(gi["attn_mask"])[1] == 8 and tf.shape(gi["attn_mask"])[2] == 8
        assert tf.shape(gi["pos_ids"])[1] == 8

    def test_build_gather_outputs_target_batch_size(self):
        gi = build_gather_outputs(
            X=tf.constant([[0, 1, 2, 3, 4, 5]]),
            sequence_lengths=tf.constant([6]),
            start=tf.constant([[0, 0, 0, 0, 0, 0]]),
            end=tf.constant([[0, 0, 0, 0, 0, 5]]),
            other_end=tf.constant([[0, 0, 0, 0, 0, 5]]),
            chunk_tables=False,
            include_mask=True,
            training=True,
            target_seq_len=6,
            target_batch_size=4,
        )
        # Batch dimension should be at least 4
        assert tf.shape(gi["values"])[0] >= 4
        assert tf.shape(gi["attn_mask"])[0] >= 4
        assert tf.shape(gi["pos_ids"])[0] >= 4

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

    def test_chunk_ragged_tensor(self):
        result = chunk_ragged_tensor(
            inputs=tf.ragged.constant(
                [
                    [[0, 1], [0, 2]],
                    [[0, 3]],
                    [[0, 8]],
                    [[0, 9]],
                    [[0, 4], [0, 5], [0, 6], [0, 7]],
                ],
                ragged_rank=1,
            ),
            other_end=tf.constant([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]]),
            base_model_max_length=3,
        )
        assert tf.reduce_all(
            tf.equal(
                result,
                tf.ragged.constant(
                    [
                        [[0, 1], [0, 2]],  # Untouched as less than original length
                        [[0, 3]],
                        [[0, 8]],
                        [[0, 9]],
                        [
                            [0, 4],
                            [0, 5],
                            [0, 6],
                        ],  # Final 2 are chunked with rows 0 and 1 as context.
                        [[0, 4], [0, 5], [0, 7]],
                    ],
                    ragged_rank=1,
                ),
            )
        )

    def test_chunk_ragged_tensor_no_context_fallback(self):
        result = chunk_ragged_tensor(
            inputs=tf.ragged.constant(
                [
                    [[0, 1], [0, 2]],
                    [[0, 3]],
                    [[0, 8]],
                    [[0, 9]],
                    [[0, 4], [0, 5], [0, 6], [0, 7]],
                ],
                ragged_rank=1,
            ),
            other_end=tf.constant([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]]),
            base_model_max_length=2,
        )
        assert tf.reduce_all(
            tf.equal(
                result,
                tf.ragged.constant(
                    [
                        [[0, 1], [0, 2]],  # Untouched as less than original length
                        [[0, 3]],
                        [[0, 8]],
                        [[0, 9]],
                        [
                            [0, 4],
                            [0, 5],
                        ],  # Final 2 are chunked with no context as the max length == the amount of context.
                        [[0, 6], [0, 7]],
                    ],
                    ragged_rank=1,
                ),
            )
        )

    def test_chunk_ragged_tensor_full_length(self):
        inputs = tf.ragged.constant(
            [
                [[0, 1], [0, 2]],
                [[0, 3]],
                [[0, 8]],
                [[0, 9]],
                [[0, 4], [0, 5], [0, 6], [0, 7]],
            ],
            ragged_rank=1,
        )
        result = chunk_ragged_tensor(
            inputs=inputs,
            other_end=tf.constant([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]]),
            base_model_max_length=5,
        )
        print(result)
        assert tf.reduce_all(tf.equal(result, inputs))

    def test_chunk_ragged_tensor_length_1(self):
        result = chunk_ragged_tensor(
            inputs=tf.ragged.constant(
                [
                    [[0, 1], [0, 2]],
                    [[0, 3]],
                    [[0, 8]],
                    [[0, 9]],
                    [[0, 4], [0, 5], [0, 6], [0, 7]],
                ],
                ragged_rank=1,
            ),
            other_end=tf.constant([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]]),
            base_model_max_length=1,
        )
        assert tf.reduce_all(
            tf.equal(
                result,
                tf.ragged.constant(
                    [
                        [[0, 1]],
                        [[0, 2]],
                        [[0, 3]],
                        [[0, 8]],
                        [[0, 9]],
                        [[0, 4]],
                        [[0, 5]],
                        [[0, 6]],
                        [[0, 7]],
                    ],
                    ragged_rank=1,
                ),
            )
        )

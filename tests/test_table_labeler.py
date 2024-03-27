import pytest
import io
from finetune.util.table_labeler import TableLabeler, TableETL
from finetune.scheduler import Scheduler


def make_labels_predictions(labels):
    for doc_labels in labels:
        for label in doc_labels:
            label["confidence"] = {label["label"]: 0.999999}
    return labels


@pytest.fixture
def labeled_table_data():
    text = [
        """Some text before the table
A B
1 2
3 4
Some text after the table"""
    ]
    labels = [
        [
            {"text": "text", "start": 5, "end": 9, "label": "non-table"},
            {"text": "A", "start": 27, "end": 28, "label": "in-table"},
            {"text": "B", "start": 29, "end": 30, "label": "in-table"},
            {"text": "text", "start": 44, "end": 48, "label": "non-table"},
        ]
    ]
    tables = [  # Documents
        [  # Pages (unused but included by OCR)
            [  # Tables
                {
                    "cells": [
                        {
                            "text": "A",
                            "rows": [0],
                            "columns": [0],
                            "doc_offsets": [{"start": 27, "end": 28}],
                        },
                        {
                            "text": "B",
                            "rows": [0],
                            "columns": [1],
                            "doc_offsets": [{"start": 29, "end": 30}],
                        },
                        {
                            "text": "1",
                            "rows": [1],
                            "columns": [0],
                            "doc_offsets": [{"start": 31, "end": 32}],
                        },
                        {
                            "text": "2",
                            "rows": [1],
                            "columns": [1],
                            "doc_offsets": [{"start": 33, "end": 34}],
                        },
                        {
                            "text": "3",
                            "rows": [2],
                            "columns": [0],
                            "doc_offsets": [{"start": 35, "end": 36}],
                        },
                        {
                            "text": "4",
                            "rows": [2],
                            "columns": [1],
                            "doc_offsets": [{"start": 37, "end": 38}],
                        },
                    ],
                    "doc_offsets": [{"start": 27, "end": 38}],
                }
            ]
        ]
    ]
    return text, labels, tables


@pytest.fixture
def labeled_table_data_split_table():
    text = [
        """A B Some
1 2 Non Table
3 4 Text"""
    ]
    labels = [
        [
            {"text": "A", "start": 0, "end": 1, "label": "in-table"},
            {"text": "B", "start": 2, "end": 3, "label": "in-table"},
            {"text": "1", "start": 9, "end": 10, "label": "in-table"},
            {"text": "2", "start": 11, "end": 12, "label": "in-table"},
            {"text": "3", "start": 23, "end": 24, "label": "in-table"},
            {"text": "4", "start": 25, "end": 26, "label": "in-table"},
        ]
    ]
    tables = [  # Documents
        [  # Pages (unused but included by OCR)
            [  # Tables
                {
                    "cells": [
                        {
                            "text": "A",
                            "rows": [0],
                            "columns": [0],
                            "doc_offsets": [{"start": 0, "end": 1}],
                        },
                        {
                            "text": "B",
                            "rows": [0],
                            "columns": [1],
                            "doc_offsets": [{"start": 2, "end": 3}],
                        },
                        {
                            "text": "1",
                            "rows": [1],
                            "columns": [0],
                            "doc_offsets": [{"start": 9, "end": 10}],
                        },
                        {
                            "text": "2",
                            "rows": [1],
                            "columns": [1],
                            "doc_offsets": [{"start": 11, "end": 12}],
                        },
                        {
                            "text": "3",
                            "rows": [2],
                            "columns": [0],
                            "doc_offsets": [{"start": 23, "end": 24}],
                        },
                        {
                            "text": "4",
                            "rows": [2],
                            "columns": [1],
                            "doc_offsets": [{"start": 25, "end": 26}],
                        },
                    ],
                    "doc_offsets": [
                        {"start": 0, "end": 3},
                        {"start": 9, "end": 12},
                        {"start": 23, "end": 26},
                    ],
                }
            ]
        ]
    ]
    return text, labels, tables


@pytest.fixture
def labeled_table_data_split_table_vertically():
    text = [
        """A B
1 2
1 2"""
    ]
    labels = [
        [
            {"text": "A", "start": 0, "end": 1, "label": "in-table"},
            {"text": "B", "start": 2, "end": 3, "label": "in-table"},
            {"text": "1", "start": 4, "end": 5, "label": "in-table"},
            {"text": "2", "start": 6, "end": 7, "label": "in-table"},
            {"text": "1", "start": 8, "end": 9, "label": "in-table"},
            {"text": "2", "start": 10, "end": 11, "label": "in-table"},
        ]
    ]
    tables = [  # Documents
        [  # Pages (unused but included by OCR)
            [  # Tables
                {
                    "cells": [
                        {
                            "text": "A",
                            "rows": [0],
                            "columns": [0],
                            "doc_offsets": [{"start": 0, "end": 1}],
                        },
                        {
                            "text": "B",
                            "rows": [0],
                            "columns": [1],
                            "doc_offsets": [{"start": 2, "end": 3}],
                        },
                        {
                            "text": "1\n1",
                            "rows": [1],
                            "columns": [0],
                            "doc_offsets": [
                                {"start": 4, "end": 5},
                                {"start": 8, "end": 9},
                            ],
                        },
                        {
                            "text": "2\n2",
                            "rows": [1],
                            "columns": [1],
                            "doc_offsets": [
                                {"start": 6, "end": 7},
                                {"start": 10, "end": 11},
                            ],
                        },
                    ],
                    "doc_offsets": [{"start": 0, "end": 11}],
                }
            ]
        ]
    ]
    return text, labels, tables


def test_fit_predict(labeled_table_data):
    filename = "tl.jl"
    text, labels, tables = labeled_table_data
    tl = TableLabeler()
    tl.fit(text=text * 10, labels=labels * 10, tables=tables * 10)
    tl.save(filename)
    del tl
    shed = Scheduler()
    preds = TableLabeler.predict_from_file(
        model_file_path=filename, text=text, tables=tables, scheduler=shed
    )
    print([{**p, "confidence": None} for p in preds[0]])
    assert len(preds[0]) == len(labels[0])
    assert set((p["start"], p["end"], p["label"]) for p in preds[0]) == set(
        (l["start"], l["end"], l["label"]) for l in labels[0]
    )


def test_fit_predict_bytes_io(labeled_table_data):
    bytes_io = io.BytesIO()
    text, labels, tables = labeled_table_data
    tl = TableLabeler()
    tl.fit(text=text * 10, labels=labels * 10, tables=tables * 10)
    tl.save(bytes_io)
    bytes_io.seek(0)
    del tl
    shed = Scheduler()
    preds = TableLabeler.predict_from_file(
        model_file_path=bytes_io, text=text, tables=tables, scheduler=shed, cache_key="test_fit_predict_bytes_io"
    )
    print([{**p, "confidence": None} for p in preds[0]])
    assert len(preds[0]) == len(labels[0])
    assert set((p["start"], p["end"], p["label"]) for p in preds[0]) == set(
        (l["start"], l["end"], l["label"]) for l in labels[0]
    )



@pytest.mark.parametrize("drop_labels", [True, False])
@pytest.mark.parametrize("inc_labels", [True, False])
def test_table_text_chunks_and_context(drop_labels, inc_labels, labeled_table_data):
    text, labels, tables = labeled_table_data
    etl = TableETL(drop_table_from_text_labels=drop_labels, chunk_tables=False)
    output = etl.get_table_text_chunks_and_context(
        text=text, tables=tables, **({"labels": labels} if inc_labels else {})
    )
    if inc_labels:
        assert output["table_labels"] == [
            [
                {"text": "A", "start": 0, "end": 1, "label": "in-table"},
                {"text": "B", "start": 2, "end": 3, "label": "in-table"},
            ]
        ]
        if drop_labels:
            assert output["doc_labels"] == [
                [
                    {"text": "text", "start": 5, "end": 9, "label": "non-table"},
                    {"text": "text", "start": 44, "end": 48, "label": "non-table"},
                ]
            ]
        else:
            assert output["doc_labels"] == [
                [
                    {"text": "text", "start": 5, "end": 9, "label": "non-table"},
                    {"text": "A", "start": 27, "end": 28, "label": "in-table"},
                    {"text": "B", "start": 29, "end": 30, "label": "in-table"},
                    {"text": "text", "start": 44, "end": 48, "label": "non-table"},
                ]
            ]
    assert output["table_text"][0] == "A B\n1 2\n3 4"
    assert output["doc_text"] == text
    assert output["table_chunks"] == [
        [{"document": {"end": 38, "start": 27}, "table": {"end": 11, "start": 0}}]
    ]
    assert output["table_context"] == [
        [
            {
                "text": "A",
                "start": 0,
                "end": 1,
                "start_row": 0,
                "end_row": 0,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "B",
                "start": 2,
                "end": 3,
                "start_row": 0,
                "end_row": 0,
                "start_col": 1,
                "end_col": 1,
            },
            {
                "text": "1",
                "start": 4,
                "end": 5,
                "start_row": 1,
                "end_row": 1,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "2",
                "start": 6,
                "end": 7,
                "start_row": 1,
                "end_row": 1,
                "start_col": 1,
                "end_col": 1,
            },
            {
                "text": "3",
                "start": 8,
                "end": 9,
                "start_row": 2,
                "end_row": 2,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "4",
                "start": 10,
                "end": 11,
                "start_row": 2,
                "end_row": 2,
                "start_col": 1,
                "end_col": 1,
            },
        ]
    ]


@pytest.mark.parametrize("drop_table_from_pred", [True, False])
def test_resolve_preds(drop_table_from_pred, labeled_table_data):
    text, labels, tables = labeled_table_data
    etl = TableETL(drop_table_from_text_preds=drop_table_from_pred, chunk_tables=False)
    resolved_preds = etl.resolve_preds(
        table_preds=make_labels_predictions(
            [[{"text": "A B", "start": 0, "end": 3, "label": "in-table"}]]
        ),
        text_preds=make_labels_predictions(
            [
                [
                    {"text": "text", "start": 5, "end": 9, "label": "non-table"},
                    {"text": "1", "start": 31, "end": 32, "label": "in-table"},
                    {"text": "text", "start": 44, "end": 48, "label": "non-table"},
                ]
            ]
        ),
        table_chunks=[
            [{"document": {"end": 38, "start": 27}, "table": {"end": 11, "start": 0}}]
        ],
        document_text=text,
        table_doc_i=[0],
        tables=tables,
    )
    if drop_table_from_pred:
        assert len(resolved_preds[0]) == 4
        assert resolved_preds == make_labels_predictions(labels)
    else:
        assert len(resolved_preds[0]) == 5
        assert resolved_preds == make_labels_predictions(
            [
                sorted(
                    labels[0]
                    + [{"text": "1", "start": 31, "end": 32, "label": "in-table"}],
                    key=lambda x: x["start"],
                )
            ]
        )


@pytest.mark.parametrize("drop_labels", [True, False])
@pytest.mark.parametrize("inc_labels", [True, False])
def test_table_text_chunks_and_context_split_horizontal(
    drop_labels, inc_labels, labeled_table_data_split_table
):
    text, labels, tables = labeled_table_data_split_table
    etl = TableETL(drop_table_from_text_labels=drop_labels, chunk_tables=False)
    output = etl.get_table_text_chunks_and_context(
        text=text, tables=tables, **({"labels": labels} if inc_labels else {})
    )
    if inc_labels:
        assert output["table_labels"] == [
            [
                {"text": "A", "start": 0, "end": 1, "label": "in-table"},
                {"text": "B", "start": 2, "end": 3, "label": "in-table"},
                {"text": "1", "start": 4, "end": 5, "label": "in-table"},
                {"text": "2", "start": 6, "end": 7, "label": "in-table"},
                {"text": "3", "start": 8, "end": 9, "label": "in-table"},
                {"text": "4", "start": 10, "end": 11, "label": "in-table"},
            ]
        ]
        if drop_labels:
            assert output["doc_labels"] == [[]]
        else:
            assert output["doc_labels"] == [
                [
                    {"text": "A", "start": 0, "end": 1, "label": "in-table"},
                    {"text": "B", "start": 2, "end": 3, "label": "in-table"},
                    {"text": "1", "start": 9, "end": 10, "label": "in-table"},
                    {"text": "2", "start": 11, "end": 12, "label": "in-table"},
                    {"text": "3", "start": 23, "end": 24, "label": "in-table"},
                    {"text": "4", "start": 25, "end": 26, "label": "in-table"},
                ]
            ]
    assert output["table_text"][0] == "A B\n1 2\n3 4"
    assert output["doc_text"] == text
    assert output["table_chunks"] == [
        [
            {"document": {"start": 0, "end": 3}, "table": {"start": 0, "end": 3}},
            {"document": {"start": 9, "end": 12}, "table": {"start": 4, "end": 7}},
            {"document": {"start": 23, "end": 26}, "table": {"start": 8, "end": 11}},
        ]
    ]
    assert output["table_context"] == [
        [
            {
                "text": "A",
                "start": 0,
                "end": 1,
                "start_row": 0,
                "end_row": 0,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "B",
                "start": 2,
                "end": 3,
                "start_row": 0,
                "end_row": 0,
                "start_col": 1,
                "end_col": 1,
            },
            {
                "text": "1",
                "start": 4,
                "end": 5,
                "start_row": 1,
                "end_row": 1,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "2",
                "start": 6,
                "end": 7,
                "start_row": 1,
                "end_row": 1,
                "start_col": 1,
                "end_col": 1,
            },
            {
                "text": "3",
                "start": 8,
                "end": 9,
                "start_row": 2,
                "end_row": 2,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "4",
                "start": 10,
                "end": 11,
                "start_row": 2,
                "end_row": 2,
                "start_col": 1,
                "end_col": 1,
            },
        ]
    ]


@pytest.mark.parametrize("drop_labels", [True, False])
@pytest.mark.parametrize("inc_labels", [True, False])
def test_table_text_chunks_and_context_split_vertical(
    drop_labels, inc_labels, labeled_table_data_split_table_vertically
):
    text, labels, tables = labeled_table_data_split_table_vertically
    etl = TableETL(drop_table_from_text_labels=drop_labels, chunk_tables=False)
    output = etl.get_table_text_chunks_and_context(
        text=text, tables=tables, **({"labels": labels} if inc_labels else {})
    )
    if inc_labels:
        assert output["table_labels"] == [
            [
                {"text": "A", "start": 0, "end": 1, "label": "in-table"},
                {"text": "B", "start": 2, "end": 3, "label": "in-table"},
                {"text": "1", "start": 4, "end": 5, "label": "in-table"},
                {"text": "2", "start": 6, "end": 7, "label": "in-table"},
                {"text": "1", "start": 8, "end": 9, "label": "in-table"},
                {"text": "2", "start": 10, "end": 11, "label": "in-table"},
            ]
        ]
        if drop_labels:
            assert output["doc_labels"] == [[]]
        else:
            assert output["doc_labels"] == [
                [
                    {"text": "A", "start": 0, "end": 1, "label": "in-table"},
                    {"text": "B", "start": 2, "end": 3, "label": "in-table"},
                    {"text": "1", "start": 4, "end": 5, "label": "in-table"},
                    {"text": "2", "start": 6, "end": 7, "label": "in-table"},
                    {"text": "1", "start": 8, "end": 9, "label": "in-table"},
                    {"text": "2", "start": 10, "end": 11, "label": "in-table"},
                ]
            ]
    assert output["table_text"][0] == "A B\n1 2\n1 2"
    assert output["doc_text"] == text
    assert output["table_chunks"] == [
        [{"document": {"start": 0, "end": 11}, "table": {"start": 0, "end": 11}}]
    ]
    assert output["table_context"] == [
        [
            {
                "text": "A",
                "start": 0,
                "end": 1,
                "start_row": 0,
                "end_row": 0,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "B",
                "start": 2,
                "end": 3,
                "start_row": 0,
                "end_row": 0,
                "start_col": 1,
                "end_col": 1,
            },
            {
                "text": "1",
                "start": 4,
                "end": 5,
                "start_row": 1,
                "end_row": 1,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "1",
                "start": 8,
                "end": 9,
                "start_row": 1,
                "end_row": 1,
                "start_col": 0,
                "end_col": 0,
            },
            {
                "text": "2",
                "start": 6,
                "end": 7,
                "start_row": 1,
                "end_row": 1,
                "start_col": 1,
                "end_col": 1,
            },
            {
                "text": "2",
                "start": 10,
                "end": 11,
                "start_row": 1,
                "end_row": 1,
                "start_col": 1,
                "end_col": 1,
            },
        ]
    ]


@pytest.mark.parametrize("drop_table_from_pred", [True, False])
def test_resolve_preds_pred_spans_table(drop_table_from_pred, labeled_table_data):
    text, labels, tables = labeled_table_data
    etl = TableETL(drop_table_from_text_preds=drop_table_from_pred, chunk_tables=False)
    resolved_preds = etl.resolve_preds(
        table_preds=make_labels_predictions(
            [[{"text": "B", "start": 2, "end": 3, "label": "in-table"}]]
        ),
        text_preds=make_labels_predictions(
            [
                [
                    {
                        "text": "Some text before the table\nA",
                        "start": 0,
                        "end": 28,
                        "label": "mixed",
                    }
                ]
            ]
        ),
        table_chunks=[
            [{"document": {"end": 38, "start": 27}, "table": {"end": 11, "start": 0}}]
        ],
        document_text=text,
        table_doc_i=[0],
        tables=tables,
    )
    if drop_table_from_pred:
        assert resolved_preds == make_labels_predictions(
            [
                [
                    {
                        "text": "Some text before the table\n",
                        "start": 0,
                        "end": 27,
                        "label": "mixed",
                    },
                    {"text": "B", "start": 29, "end": 30, "label": "in-table"},
                ]
            ]
        )
    else:
        assert resolved_preds == make_labels_predictions(
            [
                [
                    {
                        "text": "Some text before the table\n",
                        "start": 0,
                        "end": 27,
                        "label": "mixed",
                    },
                    {"text": "A", "start": 27, "end": 28, "label": "mixed"},
                    {"text": "B", "start": 29, "end": 30, "label": "in-table"},
                ]
            ]
        )


def test_subtract_spans():
    etl = TableETL(chunk_tables=False)
    result = etl.subtract_spans(
        {"start": 0, "end": 100, "some_attr": "abc"},
        [{"start": 10, "end": 20}, {"start": 50, "end": 60}],
    )
    assert result == [
        {"start": 0, "end": 10, "some_attr": "abc"},
        {"start": 20, "end": 50, "some_attr": "abc"},
        {"start": 60, "end": 100, "some_attr": "abc"},
    ]

    result = etl.subtract_spans(
        {"start": 0, "end": 100, "some_attr": "abc"}, [{"start": 0, "end": 300}]
    )
    assert result == []

    result = etl.subtract_spans(
        {"start": 0, "end": 100, "some_attr": "abc"}, [{"start": 0, "end": 10}]
    )
    assert result == [{"start": 10, "end": 100, "some_attr": "abc"}]

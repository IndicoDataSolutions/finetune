"""
Finetune-style interface for running a pipeline of table and non-table models.
"""
import copy
import functools
import os
import tempfile
import typing as t
import sys
import contextlib

from finetune.base_models import TableRoBERTa
from finetune.util.metrics import sequences_overlap
from finetune.scheduler import Scheduler
from finetune.encoding.input_encoder import BaseEncoder

from finetune import SequenceLabeler

Span = t.Dict[str, t.Any]  # Label or Pred
Table = t.Dict[str, t.Any]
PageTables = t.List[Table]
DocumentTables = t.List[PageTables]
DocumentSpans = t.List[Span]
Chunk = t.Dict[str, t.Dict[str, int]]
TableChunks = t.List[Chunk]


def overlaps_token(span, tok):
    # Otherwise 0 length tokens on the boundaries get dropped
    return sequences_overlap(span, tok) or (
        tok["end"] - tok["start"] == 0 and span["start"] <= tok["end"] <= span["end"]
    )


def _adjust_span_to_chunk(
    orig_span: Span,
    chunk: Chunk,
    output_space: str,
    input_space: str,
    output_space_text: str,
):
    span = copy.deepcopy(orig_span)
    ideal_adj_start = (
        span["start"] - chunk[input_space]["start"] + chunk[output_space]["start"]
    )
    adj_start = max(ideal_adj_start, chunk[output_space]["start"])
    adj_end = min(
        span["end"] - chunk[input_space]["start"] + chunk[output_space]["start"],
        chunk[output_space]["end"],
    )
    if "text" in span:
        span["text"] = span["text"][
            adj_start - ideal_adj_start : adj_end - ideal_adj_start
        ]
        if output_space_text is not None:
            assert span["text"] == output_space_text[adj_start:adj_end], (
                span["text"],
                output_space_text[adj_start:adj_end],
                orig_span,
                ">>",
                span,
            )
    span["start"] = adj_start
    span["end"] = adj_end
    return span


def fix_spans(
    spans: DocumentSpans,
    table_chunks: TableChunks,
    output_space_text: str,
    input_space: str,
    output_space: str,
) -> DocumentSpans:
    """
    convers a list of spans (anything with a start, end and optionally text, usually labels or context)
    between 2 character spaces (document and table)

    Will do 1 -> many splits when a span spreads across multiple chunks.

    Args:
        spans: The spans (labels or context) to move between the character spaces.
        chunk: The chunks a list of dicts including input_space and output_space keys.
        output_space_text: The text of the output space, used for validation.
        input_space: a key into chunks of the space the current spans are in.
        output_space: a key into chunks of the space to move the current spans
        strict: bool, whether to assert that every span is matched by at least one chunk.

    """
    spans_out = []
    for span in spans:
        for table_chunk in table_chunks:
            if sequences_overlap(span, table_chunk[input_space]):
                spans_out.append(
                    _adjust_span_to_chunk(
                        span, table_chunk, output_space, input_space, output_space_text
                    )
                )
    return spans_out


class TableETL:
    # Overrides that maintain backwards compatibility when loading old models.
    BACKWARDS_COMPAT_OVERRIDES = {"split_preds_to_cells": False, "chunk_tables": False}

    def __init__(
        self,
        drop_table_from_text_labels: bool = False,
        drop_table_from_text_preds: bool = True,
        split_preds_to_cells: bool = True,
        chunk_tables: bool = True,
    ):
        self.drop_table_from_text_labels = drop_table_from_text_labels
        self.drop_table_from_text_preds = drop_table_from_text_preds
        self.split_preds_to_cells = split_preds_to_cells
        self.chunk_tables = chunk_tables

    def __setstate__(self, state):
        # A simple way so that we always pick up new default settings.
        self.__dict__.update({**self.BACKWARDS_COMPAT_OVERRIDES, **state})

    def get_table_text(self, text: str, doc_offsets: t.List[t.Dict[str, int]]) -> str:
        for c in doc_offsets:
            assert c["end"] <= len(text), "Table offsets must exist within document."
        return "\n".join(text[c["start"] : c["end"]] for c in doc_offsets)

    def get_table_context(
        self,
        doc_text: str,
        table_text: str,
        table: t.Dict[str, t.Any],
        table_chunks: TableChunks,
    ):
        unadjusted_context = []
        for cell in table["cells"]:
            for doc_offset in cell["doc_offsets"]:
                unadjusted_context.append(
                    {
                        "text": doc_text[doc_offset["start"] : doc_offset["end"]],
                        "start": doc_offset["start"],
                        "end": doc_offset["end"],
                        "start_row": min(cell["rows"]),
                        "end_row": max(cell["rows"]),
                        "start_col": min(cell["columns"]),
                        "end_col": max(cell["columns"]),
                    }
                )
        return fix_spans(
            spans=unadjusted_context,
            table_chunks=table_chunks,
            output_space_text=table_text,
            input_space="document",
            output_space="table",
        )

    def create_chunks_from_doc_offset(
        self, doc_offsets: t.List[t.Dict[str, int]]
    ) -> TableChunks:
        """
        Converts the offet list from the table into a Chunk object.
        Storing the mapping from original document to the table-text
        as output by get_table_text.
        """
        output = []
        for offset in doc_offsets:
            c_start = (
                0 if not output else output[-1]["table"]["end"] + 1
            )  # +1 to account for us joining on newlines.
            output.append(
                {
                    "document": {"start": offset["start"], "end": offset["end"]},
                    "table": {
                        "start": c_start,
                        "end": c_start + offset["end"] - offset["start"],
                    },
                }
            )
        return output

    def subtract_spans(
        self, input_span: Span, to_remove: t.List[t.Dict[str, t.Any]]
    ) -> t.List[Span]:
        """
        If any item in to_remove overlaps with input span it is removed from the span
        creating between 0 and to_remove + 1 spans as output.

        For example
        {"start" 0, "end": 10} - [{"start": 0, "end": 2}, {"start": 5, "end": 6}]
        = [{"start": 2, "end": 5}, {"start": 6, "end": 10}]
        """
        output = [input_span]

        for remove_span in to_remove:
            new_output = []

            for span in output:
                if (
                    remove_span["end"] <= span["start"]
                    or remove_span["start"] >= span["end"]
                ):
                    new_output.append(span)
                else:
                    if remove_span["start"] > span["start"]:
                        new_output.append({**span, "end": remove_span["start"]})
                    if remove_span["end"] < span["end"]:
                        new_output.append({**span, "start": remove_span["end"]})

            output = new_output

        return output

    def remove_table_labels(
        self, doc_labels: DocumentSpans, doc_chunks: t.List[TableChunks], doc_text: str
    ):
        flat_doc_chunks = [ts["document"] for chunks in doc_chunks for ts in chunks]
        output_labels = []
        for l in doc_labels:
            if not any(sequences_overlap(l, ts) for ts in flat_doc_chunks):
                output_labels.append(l)
            else:
                sub_spans = self.subtract_spans(l, flat_doc_chunks)
                for s in sub_spans:
                    s["text"] = doc_text[s["start"] : s["end"]]
                output_labels += sub_spans

        return output_labels

    def get_table_text_chunks_and_context(
        self,
        text: t.List[str],
        tables: t.List[DocumentTables],
        labels: t.List[DocumentSpans] = None,
    ) -> t.Dict[str, dict]:
        # The following items are 1 entry per-table.
        table_chunks_output = []  # Chunk objects
        table_context_output = []  # Context, formatted for tinetune
        table_doc_index_output = []  # The original document index for a table.
        table_text_output = []  # Text of the table only.
        table_labels_output = (
            []
        )  # Labels of the table only. With start and end to match the text

        # text_labels_output is 1 item per document.
        text_labels_output = []

        for i, (document_text, document_tables, document_labels) in enumerate(
            zip(text, tables, labels if labels else [None] * len(text))
        ):
            # These lists are to accumulate table labels and table chunks for a given document.
            document_table_labels = []
            document_table_chunks = []

            for page_tables in document_tables:
                for table in page_tables:
                    table_doc_offsets = table["doc_offsets"]
                    table_text_i = self.get_table_text(document_text, table_doc_offsets)
                    if table_text_i.strip() == "":
                        # Seems like the table model is occasionally outputting empty tables.
                        continue
                    table_doc_index_output.append(i)
                    table_text_output.append(table_text_i)
                    table_chunks_i = self.create_chunks_from_doc_offset(
                        table_doc_offsets
                    )
                    document_table_chunks.append(table_chunks_i)
                    table_context_output.append(
                        self.get_table_context(
                            doc_text=document_text,
                            table_text=table_text_i,
                            table=table,
                            table_chunks=table_chunks_i,
                        )
                    )
                    if document_labels is not None:
                        document_table_labels.append(
                            fix_spans(
                                document_labels,
                                table_chunks=table_chunks_i,
                                output_space_text=table_text_i,
                                input_space="document",
                                output_space="table",
                            )
                        )
            table_labels_output += document_table_labels
            table_chunks_output += document_table_chunks

            if self.drop_table_from_text_labels and labels is not None:
                text_labels_output.append(
                    self.remove_table_labels(
                        doc_labels=document_labels,
                        doc_chunks=document_table_chunks,
                        doc_text=document_text,
                    )
                )
            else:
                text_labels_output.append(document_labels)
        if labels is not None:
            output = {
                "doc_labels": text_labels_output,
                "table_labels": table_labels_output,
            }
        else:
            output = dict()

        return {
            **output,
            "table_text": table_text_output,
            "table_chunks": table_chunks_output,
            "table_context": table_context_output,
            "table_doc_i": table_doc_index_output,
            "doc_text": text,
        }

    def cleanup_predictions(
        self, predictions: DocumentSpans, tables: DocumentTables, doc_text: str
    ):
        """Cleans up predictions, applying deduplication and optionally splitting on cell boundaries.

        Args:
            predictions (DocumentSpans): Resolved model predictions.
            tables (DocumentTables): document tables
            doc_text (str): the text of the document.

        Returns:
            DocumentSpans: clean predictions
        """
        if self.split_preds_to_cells:
            cell_bounded_predictions = []
            for pred in predictions:
                cell_bounds = []
                for page_tables in tables:
                    for table in page_tables:
                        if not any(
                            sequences_overlap(do, pred) for do in table["doc_offsets"]
                        ):
                            # Optimization
                            continue
                        for cell in table["cells"]:
                            for do in cell["doc_offsets"]:
                                if sequences_overlap(do, pred):
                                    cell_bounds.append(do)
                for do in cell_bounds:
                    cell_bounded_predictions.append({**pred, **do})
                # Remove any spans that overlap the cells.
                cell_bounded_predictions += self.subtract_spans(pred, cell_bounds)
            predictions = cell_bounded_predictions

        # Deduplicate / resolve overlap - Primarily comes from chunking.
        deduplicated_preds = []
        # Sorting from most confident to least confident so that we prioritise the most confident.
        for pred in sorted(predictions, key=lambda x: -(x["confidence"][x["label"]])):
            # After subtract spans the text and bounds can disagree.
            pred["text"] = doc_text[pred["start"] : pred["end"]]
            # And we can also get whitespace only predictions from the bits between cells.
            if pred["text"].strip() == "":
                continue
            if not any(sequences_overlap(pred, ddp) for ddp in deduplicated_preds):
                deduplicated_preds.append(pred)
        return sorted(deduplicated_preds, key=lambda x: x["start"])

    def resolve_preds(
        self,
        table_preds: t.List[DocumentSpans],
        text_preds: t.List[DocumentSpans],
        table_chunks: t.List[TableChunks],
        document_text: t.List[str],
        table_doc_i: t.List[int],
        tables: t.List[DocumentTables],
    ) -> t.List[DocumentSpans]:
        output_preds = []

        # First sort the table preds and chunks back into a list that aligns with documents
        table_pred_chunks = [[] for _ in document_text]
        for p, c, i in zip(table_preds, table_chunks, table_doc_i):
            table_pred_chunks[i].append((p, c))

        for i, (text_preds_i, document_text_i, doc_pred_chunks) in enumerate(
            zip(text_preds, document_text, table_pred_chunks)
        ):
            output_doc_preds = []
            for table_preds, table_chunks in doc_pred_chunks:
                output_doc_preds += fix_spans(
                    table_preds,
                    table_chunks=table_chunks,
                    output_space_text=document_text_i,
                    input_space="chunk" if self.chunk_tables else "table",
                    output_space="document",
                )
            if self.drop_table_from_text_preds:
                output_doc_preds += self.remove_table_labels(
                    text_preds_i,
                    [d[1] for d in doc_pred_chunks],
                    doc_text=document_text_i,
                )
            else:
                output_doc_preds += text_preds_i
            output_preds.append(
                self.cleanup_predictions(
                    output_doc_preds, tables=tables[i], doc_text=document_text[i]
                )
            )
        return output_preds


@functools.lru_cache(maxsize=None)
def get_etl_from_file(model_file_path):
    return SequenceLabeler.load(model_file_path, key="etl")


class TableChunker:
    """
    Handles chunking by adding a new set of offsets system keyed by "chunk"
    Performs chunking based on the following rules

    * Select longest axis of the table, rows or columns
    * For this longest axis, chunk such that each chunk's effective num tokens is less than the defined max_length
    * effective num tokens is the number of tokens you get if you duplicate tokens by the number of cells they span.
    * Each chunk get's n_rows_context rows of context from the beginning of the opposite axis to the one being chunked (2 rows of context if columns are chunked)
    * If n_rows_context takes up more than max_length / 2 effective tokens then this context is backed off until the amount of context is less than max_length / 2.


    There is no invese chunking as TableETL.resolve_preds will simply map directly from chunk to document if chunking is enabled.
    """

    def __init__(
        self, max_length: str, tokenizer: BaseEncoder, n_rows_context: int = 2
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.n_rows_context = n_rows_context

    def get_axis_spans(self, context, token_bounds, context_key):
        max_row = max(r[context_key] for r in context)
        row_spans = [
            [
                {
                    "start": c["start"],
                    "end": c["end"],
                    "max_cell_span": max(
                        c["end_col"] - c["start_col"], c["end_row"] - c["start_row"]
                    )
                    + 1,
                }
                for c in context
                if i == c[context_key]
            ]
            for i in range(max_row + 1)
        ]
        assert sum(len(rs) for rs in row_spans) == len(context)
        return self.combine_row_spans(row_spans, token_bounds)

    def chunk(self, table_text_chunks_and_context):
        texts = []
        if "table_labels" in table_text_chunks_and_context:
            labels = []
        else:
            labels = None
        contexts = []
        doc_is = []
        chunks = []

        for text, label, context, doc_i, chunk in zip(
            table_text_chunks_and_context["table_text"],
            table_text_chunks_and_context.get(
                "table_labels",
                [None] * len(table_text_chunks_and_context["table_text"]),
            ),
            table_text_chunks_and_context["table_context"],
            table_text_chunks_and_context["table_doc_i"],
            table_text_chunks_and_context["table_chunks"],
        ):
            tokens = self.tokenizer.encode_multi_input(
                [text],
                max_length=sys.maxsize,
                remove_repeated_whitespace=True,
                include_bos_eos=False,
            )
            token_bounds = [
                {"start": s, "end": e}
                for s, e in zip(tokens.token_starts, tokens.token_ends)
            ]
            combined_row_spans = self.get_axis_spans(
                context=context, token_bounds=token_bounds, context_key="start_row"
            )
            combined_col_spans = self.get_axis_spans(
                context=context, token_bounds=token_bounds, context_key="start_col"
            )
            if max(r["num_effective_tokens"] for r in combined_row_spans) > max(
                c["num_effective_tokens"] for c in combined_col_spans
            ):
                # Split along the narrowest axis.
                combined_axis_spans = combined_col_spans
            else:
                combined_axis_spans = combined_row_spans

            spans = self._make_chunks(combined_axis_spans)
            for chunk_spans in spans:
                new_chunk_mapping = self._create_chunk_mappings(chunk, chunk_spans)
                if len(chunk_spans) == 0 or len(new_chunk_mapping) == 0:
                    continue
                chunk_text = "\n".join(
                    text[c["table"]["start"] : c["table"]["end"]]
                    for c in new_chunk_mapping
                )
                texts.append(chunk_text)
                if labels is not None:
                    labels.append(
                        fix_spans(
                            label,
                            new_chunk_mapping,
                            output_space="chunk",
                            input_space="table",
                            output_space_text=chunk_text,
                        )
                    )
                contexts.append(
                    fix_spans(
                        context,
                        new_chunk_mapping,
                        output_space="chunk",
                        input_space="table",
                        output_space_text=chunk_text,
                    )
                )
                doc_is.append(doc_i)
                chunks.append(new_chunk_mapping)
        if labels is not None:
            label_output = {"table_labels": labels}
        else:
            label_output = dict()

        return {
            **table_text_chunks_and_context,
            **label_output,
            "table_text": texts,
            "table_chunks": chunks,
            "table_context": contexts,
            "table_doc_i": doc_is,
        }

    def _create_chunk_mappings(self, original_chunks, new_chunk_points):
        chunk_outputs = []
        original_chunks = sorted(original_chunks, key=lambda x: x["table"]["start"])
        new_chunk_offset = 0
        for chunk_span in new_chunk_points:
            for og_chunk in original_chunks:
                if sequences_overlap(og_chunk["table"], chunk_span):
                    relevant_start = max(
                        og_chunk["table"]["start"], chunk_span["start"]
                    )
                    offset_to_table = relevant_start - og_chunk["table"]["start"]
                    relevant_end = min(og_chunk["table"]["end"], chunk_span["end"])
                    relevant_length = relevant_end - relevant_start
                    chunk_outputs.append(
                        {
                            "document": {
                                "start": og_chunk["document"]["start"]
                                + offset_to_table,
                                "end": og_chunk["document"]["start"]
                                + offset_to_table
                                + relevant_length,
                            },
                            "table": {
                                "start": og_chunk["table"]["start"] + offset_to_table,
                                "end": og_chunk["table"]["start"]
                                + offset_to_table
                                + relevant_length,
                            },
                            "chunk": {
                                "start": new_chunk_offset,
                                "end": new_chunk_offset + relevant_length,
                            },
                        }
                    )
                    new_chunk_offset += relevant_length + 1
        return chunk_outputs

    def _num_tokens(self, rows, effective_tokens=True):
        if effective_tokens:
            return sum(c["num_effective_tokens"] for c in rows)
        return sum(c["num_tokens"] for c in rows)

    def _make_chunks(self, row_spans):
        for n_rows_context in range(self.n_rows_context, -1, -1):
            context = row_spans[:n_rows_context]
            context_tokens = self._num_tokens(context)
            if context_tokens < (self.max_length // 2):
                break
        max_len_chunks = []
        temp_rows = []
        context_included = False
        for row in row_spans[n_rows_context:]:
            if self._num_tokens(context + temp_rows + [row]) < self.max_length:
                temp_rows.append(row)
            elif len(temp_rows) == 0:
                # The current row is too long to use any context at all.
                max_len_chunks.append([row])
            else:
                context_included = True
                max_len_chunks.append(copy.deepcopy(context) + temp_rows)
                temp_rows = [row]
        if temp_rows or not context_included:
            max_len_chunks.append(copy.deepcopy(context) + temp_rows)
        output_spans = []
        for chunk in max_len_chunks:
            temp_chunk = []
            for span in sorted(
                (span for row in chunk for span in row["spans"]),
                key=lambda x: x["start"],
            ):
                if temp_chunk and temp_chunk[-1]["end"] + 1 == span["start"]:
                    temp_chunk[-1]["end"] = span["end"]
                    temp_chunk[-1]["num_tokens"] += span["num_tokens"]
                    temp_chunk[-1]["num_effective_tokens"] += span[
                        "num_effective_tokens"
                    ]
                else:
                    temp_chunk.append(span)
            output_spans.append(temp_chunk)
        return output_spans

    def combine_row_spans(self, row_spans, token_spans):
        def mark_token(t):
            t["used"] = True
            return t

        total_num_tokens = 0
        combined_rows = []
        for row in row_spans:
            row_out = []
            for span in sorted(row, key=lambda x: x["start"]):
                if row_out and row_out[-1]["end"] + 1 == span["start"]:
                    row_out[-1]["end"] = span["end"]
                else:
                    row_out.append(span)
            for row_span in row_out:
                row_span["num_tokens"] = len(
                    [mark_token(t) for t in token_spans if overlaps_token(row_span, t)]
                )
                # Accounts for the fact that cells are duplicated when they span cells.
                row_span["num_effective_tokens"] = (
                    row_span["num_tokens"] * row_span["max_cell_span"]
                )
                total_num_tokens += row_span["num_tokens"]
            combined_rows.append(
                {
                    "num_tokens": sum(r["num_tokens"] for r in row_out),
                    "num_effective_tokens": sum(
                        r["num_effective_tokens"] for r in row_out
                    ),
                    "spans": row_out,
                }
            )
        assert len([t for t in token_spans if not t.get("used", False)]) == 0
        return combined_rows

    @classmethod
    def from_table_model(cls, table_model):
        return TableChunker(
            max_length=table_model.config.max_length,
            tokenizer=table_model.input_pipeline.text_encoder,
        )


class TableLabeler:
    def __init__(
        self,
        table_model_config: t.Dict[str, t.Any] = None,
        text_model_config: t.Dict[str, t.Any] = None,
        drop_table_from_text_labels: bool = False,
        drop_table_from_text_preds: bool = True,
        split_preds_to_cells: bool = True,
        chunk_tables: bool = True,
    ):
        self.etl = TableETL(
            drop_table_from_text_labels=drop_table_from_text_labels,
            drop_table_from_text_preds=drop_table_from_text_preds,
            split_preds_to_cells=split_preds_to_cells,
            chunk_tables=chunk_tables,
        )
        # Delay construction of these models.
        self._get_table_model = functools.partial(
            SequenceLabeler, base_model=TableRoBERTa, **(table_model_config or {})
        )
        self._get_text_model = functools.partial(
            SequenceLabeler, **(text_model_config or {})
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.text_model_path = os.path.join(self.temp_dir.name, "text.jl")
        self.table_model_path = os.path.join(self.temp_dir.name, "table.jl")

    def _fit_table_model(self, model_inputs):
        table_model = self._get_table_model()
        if self.etl.chunk_tables:
            model_inputs = TableChunker.from_table_model(table_model).chunk(
                model_inputs
            )
        table_model.fit(
            model_inputs["table_text"],
            model_inputs["table_labels"],
            context=model_inputs["table_context"],
        )
        table_model.save(self.table_model_path)
        return model_inputs

    def _fit_text_model(self, model_inputs):
        text_model = self._get_text_model()
        text_model.fit(model_inputs["doc_text"], model_inputs["doc_labels"])
        text_model.save(self.text_model_path)

    def fit(
        self,
        text: t.List[str],
        tables: t.List[DocumentTables],
        labels: t.List[DocumentSpans],
    ):
        model_inputs = self.etl.get_table_text_chunks_and_context(
            text=text, tables=tables, labels=labels
        )
        model_inputs = self._fit_table_model(model_inputs)
        self._fit_text_model(model_inputs)

    def save(self, path: str) -> None:
        with open(self.text_model_path, "rb") as text_model, open(
            self.table_model_path, "rb"
        ) as table_model:
            SequenceLabeler.save_multiple(
                path,
                models={
                    "table": table_model,
                    "text": text_model,
                    "etl": self.etl,
                },
            )

    def predict(self, text, tables, model_path=None, **kwargs):
        # Please don't use this in production
        # it's just a shim for predict_from_file so we don't have 2 separate predict implementations.
        scheduler = Scheduler()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = model_path or os.path.join(tmpdir, "table_model.jl")
            self.save(model_path)
            return self.predict_from_file(
                text=text,
                tables=tables,
                model_file_path=model_path,
                scheduler=scheduler,
                **kwargs
            )

    @classmethod
    def predict_from_file(
        cls,
        text: t.List[str],
        tables: t.List[DocumentTables],
        model_file_path: str,
        scheduler: Scheduler,
        return_negative_confidence=False,
        config_overrides=None,
        cache_key=None,
        table_model_config_overrides=None,
    ):
        etl = scheduler.load_etl(model_file_path, cache_key=cache_key)
        model_inputs = etl.get_table_text_chunks_and_context(text=text, tables=tables)
        table_model = scheduler.get_model(
            model_file_path,
            key="table",
            config_overrides=table_model_config_overrides,
            cache_key=cache_key,
        )
        if etl.chunk_tables:
            model_inputs = TableChunker.from_table_model(table_model).chunk(
                model_inputs
            )
        table_preds = scheduler.predict(
            model_file_path,
            model_inputs["table_text"],
            context=model_inputs["table_context"],
            key="table",
            return_negative_confidence=return_negative_confidence,
            cache_key=cache_key,
            config_overrides=table_model_config_overrides,
        )
        text_preds = scheduler.predict(
            model_file_path,
            model_inputs["doc_text"],
            key="text",
            config_overrides=config_overrides,
            return_negative_confidence=return_negative_confidence,
            cache_key=cache_key,
        )

        return etl.resolve_preds(
            table_preds=table_preds,
            text_preds=text_preds,
            table_chunks=model_inputs["table_chunks"],
            document_text=model_inputs["doc_text"],
            table_doc_i=model_inputs["table_doc_i"],
            tables=tables,
        )

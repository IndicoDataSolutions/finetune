"""
Finetune-style interface for running a pipeline of table and non-table models.
"""
import copy
import functools
import typing as t

from finetune.base_models import TableRoBERTa, RoBERTa
from finetune.util.metrics import sequences_overlap
from finetune.scheduler import Scheduler

from finetune import SequenceLabeler

Span = t.Dict[str, t.Any]  # Label or Pred
Table = t.Dict[str, t.Any]
PageTables = t.List[Table]
DocumentTables = t.List[PageTables]
DocumentSpans = t.List[Span]
Chunk = t.Dict[str, t.Dict[str, int]]
TableChunks = t.List[Chunk]


class TableETL:
    def __init__(
        self,
        drop_table_from_text_labels: bool = False,
        drop_table_from_text_preds: bool = True,
    ):
        self.drop_table_from_text_labels = drop_table_from_text_labels
        self.drop_table_from_text_preds = drop_table_from_text_preds

    def get_table_text(self, text: str, doc_offsets: t.List[t.Dict[str, int]]) -> str:
        return "\n".join(text[c["start"] : c["end"]] for c in doc_offsets)

    def _adjust_span_to_chunk(
        self,
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
        self,
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
                        self._adjust_span_to_chunk(
                            span,
                            table_chunk,
                            output_space,
                            input_space,
                            output_space_text,
                        )
                    )
        return spans_out

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
        return self.fix_spans(
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
        assert isinstance(doc_chunks, list)
        assert isinstance(doc_chunks[0], list)
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
                    table_doc_index_output.append(i)
                    table_text_i = self.get_table_text(document_text, table_doc_offsets)
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
                            self.fix_spans(
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

    def resolve_preds(
        self,
        table_preds: t.List[DocumentSpans],
        text_preds: t.List[DocumentSpans],
        table_chunks: t.List[TableChunks],
        document_text: t.List[str],
        table_doc_i: t.List[int],
    ) -> t.List[DocumentSpans]:
        output_preds = []

        # First sort the table preds and chunks back into a list that aligns with documents
        table_pred_chunks = [[] for _ in document_text]
        for p, c, i in zip(table_preds, table_chunks, table_doc_i):
            table_pred_chunks[i].append((p, c))

        for (i, (text_preds_i, document_text_i, doc_pred_chunks)) in enumerate(
            zip(text_preds, document_text, table_pred_chunks)
        ):
            output_doc_preds = []
            for table_preds, table_chunks in doc_pred_chunks:
                output_doc_preds += self.fix_spans(
                    table_preds,
                    table_chunks=table_chunks,
                    output_space_text=document_text_i,
                    input_space="table",
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
            output_preds.append(sorted(output_doc_preds, key=lambda x: x["start"]))
        return output_preds


@functools.lru_cache
def get_etl_from_file(model_file_path):
    return SequenceLabeler.load(model_file_path, key="etl")


class TableLabeler:
    def __init__(
        self,
        table_model_config: t.Dict[str, t.Any] = None,
        text_model_config: t.Dict[str, t.Any] = None,
        drop_table_from_text_labels: bool = False,
        drop_table_from_text_preds: bool = True,
    ):
        self.etl = TableETL(
            drop_table_from_text_labels=drop_table_from_text_labels,
            drop_table_from_text_preds=drop_table_from_text_preds,
        )
        self.table_model = SequenceLabeler(
            base_model=TableRoBERTa, **(table_model_config or {})
        )
        self.text_model = SequenceLabeler(**(text_model_config or {}))

    def fit(
        self,
        text: t.List[str],
        tables: t.List[DocumentTables],
        labels: t.List[DocumentSpans],
    ):
        model_inputs = self.etl.get_table_text_chunks_and_context(
            text=text, tables=tables, labels=labels
        )
        self.table_model.fit(
            model_inputs["table_text"],
            model_inputs["table_labels"],
            context=model_inputs["table_context"],
        )
        self.text_model.fit(model_inputs["doc_text"], model_inputs["doc_labels"])

    def save(self, path: str) -> None:
        SequenceLabeler.save_multiple(
            path,
            models={
                "table": self.table_model,
                "text": self.text_model,
                "etl": self.etl,
            },
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
    ):
        etl = get_etl_from_file(model_file_path)
        model_inputs = etl.get_table_text_chunks_and_context(text=text, tables=tables)
        table_preds = scheduler.predict(
            model_file_path,
            model_inputs["table_text"],
            context=model_inputs["table_context"],
            key="table",
            return_negative_confidence=return_negative_confidence,
        )
        text_preds = scheduler.predict(
            model_file_path,
            model_inputs["doc_text"],
            key="text",
            config_overrides=config_overrides,
            return_negative_confidence=return_negative_confidence,
        )
        return etl.resolve_preds(
            table_preds=table_preds,
            text_preds=text_preds,
            table_chunks=model_inputs["table_chunks"],
            document_text=model_inputs["doc_text"],
            table_doc_i=model_inputs["table_doc_i"],
        )

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

    def fix_spans(
        self,
        spans: t.List[t.Dict[str, t.Any]],
        chunk: t.List[t.Dict[str, t.Dict[str, int]]],
        output_space_text: str,
        input_space: str,
        output_space: str,
        strict: bool = False,
    ) -> t.List[t.Dict[str, t.Any]]:
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
        for orig_span in spans:
            matched_chunk = False
            for sub_chunk in chunk:
                if sequences_overlap(orig_span, sub_chunk[input_space]):
                    matched_chunk = True
                    span = copy.deepcopy(orig_span)
                    ideal_adj_start = (
                        span["start"]
                        - sub_chunk[input_space]["start"]
                        + sub_chunk[output_space]["start"]
                    )
                    adj_start = max(ideal_adj_start, sub_chunk[output_space]["start"])
                    adj_end = min(
                        span["end"]
                        - sub_chunk[input_space]["start"]
                        + sub_chunk[output_space]["start"],
                        sub_chunk[output_space]["end"],
                    )
                    span["start"] = max(
                        span["start"]
                        - sub_chunk[input_space]["start"]
                        + sub_chunk[output_space]["start"],
                        0,
                    )
                    span["end"] = min(
                        span["end"]
                        - sub_chunk[input_space]["start"]
                        + sub_chunk[output_space]["start"],
                        sub_chunk[output_space]["end"],
                    )
                    if "text" in span:
                        span["text"] = span["text"][
                            adj_start - ideal_adj_start : adj_end - ideal_adj_start
                        ]
                        if output_space_text is not None:
                            assert (
                                span["text"] == output_space_text[adj_start:adj_end]
                            ), (
                                span["text"],
                                output_space_text[adj_start:adj_end],
                                orig_span,
                                ">>",
                                span,
                            )
                    span["start"] = adj_start
                    span["end"] = adj_end
                    spans_out.append(span)
            if strict:
                assert matched_chunk, (
                    span,
                    [c[input_space] for c in chunk["sub_chunks"]],
                    [c[output_space] for c in chunk["sub_chunks"]],
                )
        return spans_out

    def get_table_context(self, doc_text, table_text, table, chunk):
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
            chunk=chunk,
            output_space_text=table_text,
            input_space="document",
            output_space="table",
        )

    def create_chunks_from_doc_offset(self, doc_offsets):
        output = []
        for offset in doc_offsets:
            c_start = (
                0 if not output else output[-1]["table"]["end"] + 1
            )  # +1 to account for us joining on newlines.
            output.append(
                {
                    "document": {"start": offset["start"], "end": offset["end"]},
                    "table": {"start": c_start, "end": c_start + offset["end"] - offset["start"]},
                }
            )
        return output

    def remove_table_labels(self, doc_labels, doc_chunks):
        assert isinstance(doc_chunks, list)
        assert isinstance(doc_chunks[0], list)
        output_labels = []
        for l in doc_labels:
            if not any(
                sequences_overlap(l, ts["document"])
                for chunks in doc_chunks
                for ts in chunks
            ):
                output_labels.append(l)
        return output_labels

    def get_table_text_chunks_and_context(
        self,
        text: t.List[str],
        tables: t.List[t.List[t.Dict[str, t.Any]]],
        labels: t.List[t.List[t.Dict[str, t.Any]]] = None,
    ) -> t.Dict[str, dict]:
        table_text = []
        table_chunks = []
        table_context = []
        table_doc_i = []
        table_labels = []
        doc_labels = []
        doc_text = []
        for i, (text_i, tables_i, labels_i) in enumerate(
            zip(text, tables, labels if labels else [None] * len(text))
        ):
            doc_table_labels = []
            doc_chunks = []
            for page_tables in tables_i:
                for table in page_tables:
                    table_doc_offsets = table["doc_offsets"]
                    table_doc_i.append(i)
                    table_text_i = self.get_table_text(text_i, table_doc_offsets)
                    table_text.append(table_text_i)
                    table_chunks_i = self.create_chunks_from_doc_offset(
                        table_doc_offsets
                    )
                    doc_chunks.append(table_chunks_i)
                    table_context.append(
                        self.get_table_context(
                            doc_text=text_i,
                            table_text=table_text_i,
                            table=table,
                            chunk=table_chunks_i,
                        )
                    )
                    if labels_i is not None:
                        doc_table_labels.append(
                            self.fix_spans(
                                labels_i,
                                chunk=table_chunks_i,
                                output_space_text=table_text_i,
                                input_space="document",
                                output_space="table",
                            )
                        )
            table_labels += doc_table_labels
            table_chunks += doc_chunks
            if self.drop_table_from_text_labels and labels is not None:
                doc_labels.append(
                    self.remove_table_labels(doc_labels=labels_i, doc_chunks=doc_chunks)
                )
            else:
                doc_labels.append(labels_i)
            doc_text.append(text_i)
        if labels is not None:
            output = {"doc_labels": doc_labels, "table_labels": table_labels}
        else:
            output = dict()

        return {
            **output,
            "table_text": table_text,
            "table_chunks": doc_chunks,
            "table_context": table_context,
            "table_doc_i": table_doc_i,
            "doc_text": doc_text,
        }

    def resolve_preds(
        self, table_preds, text_preds, table_chunks, document_text, table_doc_i
    ):
        output_preds = []
        table_pred_chunks = [[] for _ in document_text]
        for p, c, i in zip(table_preds, table_chunks, table_doc_i):
            table_pred_chunks[i].append((p, c))

        for (i, (text_preds_i, document_text_i, doc_pred_chunks)) in enumerate(
            zip(text_preds, document_text, table_pred_chunks)
        ):
            output_doc_preds = []
            for p, c in doc_pred_chunks:
                output_doc_preds += self.fix_spans(
                    p,
                    chunk=c,
                    output_space_text=document_text_i,
                    input_space="table",
                    output_space="document",
                )
            if self.drop_table_from_text_preds:
                output_doc_preds += self.remove_table_labels(
                    text_preds_i, [d[1] for d in doc_pred_chunks]
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
        normal_model_config: t.Dict[str, t.Any] = None,
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
        self.text_model = SequenceLabeler(
            base_model=RoBERTa, **(normal_model_config or {})
        )

    def fit(
        self,
        text: t.List[str],
        tables: t.List[t.List[t.Dict[str, t.Any]]],
        labels: t.List[t.List[t.Dict[str, t.Any]]],
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

    def save(self, path: str):
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
        tables: t.List[t.List[t.Dict[str, t.Any]]],
        model_file_path: str,
        scheduler: Scheduler,
    ):
        etl = get_etl_from_file(model_file_path)
        model_inputs = etl.get_table_text_chunks_and_context(text=text, tables=tables)
        table_preds = scheduler.predict(
            model_file_path,
            model_inputs["table_text"],
            context=model_inputs["table_context"],
            key="table",
        )
        text_preds = scheduler.predict(
            model_file_path, model_inputs["doc_text"], key="text"
        )
        return etl.resolve_preds(
            table_preds=table_preds,
            text_preds=text_preds,
            table_chunks=model_inputs["table_chunks"],
            document_text=model_inputs["doc_text"],
            table_doc_i=model_inputs["table_doc_i"],
        )

import warnings

from finetune.target_models.sequence_labeling import SequenceLabeler, SequencePipeline
from finetune.encoding.input_encoder import EncodedOutput
from finetune.errors import FinetuneError
from finetune.base_models import DocRep, LayoutLM

def get_context(document, dpi_norm, base_model):
    """
    Gets the context as formatted for the DocRep base model from 
    the output of indico's PDFExtraction api.
    """
    if base_model == DocRep:
        return get_context_doc_rep(document, dpi_norm)
    elif base_model == LayoutLM:
        return get_context_layoutlm(document)
    else:
        warnings.warn("Running DocumentLabeler with a base model that doesn't utilize position info.")
        return get_context_doc_rep(document, dpi_norm)

def get_context_layoutlm(document):
    context = []
    for page in document:
        for token in page["tokens"]:
            pos = token["position"]
            offset = token["doc_offset"]
            width = page["pages"][0]["size"]["width"]
            height = page["pages"][0]["size"]["height"]
            context.append(
                {
                    'top': int(pos["top"] / height * 1000),
                    'bottom': int(pos["bottom"] / height * 1000),
                    'left': int(pos["left"] / width * 1000),
                    'right': int(pos["right"] / width * 1000),
                    'start': offset["start"],
                    'end': offset["end"],
                    'text': token["text"],
                }
            )
    return context

def get_context_doc_rep(document, dpi_norm):
    context = []
    for page in document:
        # This norm factor is 
        if dpi_norm:
            dpi = page["pages"][0]["dpi"]
            x_norm = 300 / dpi["dpix"]
            y_norm = 300 / dpi["dpiy"]
        else:
            x_norm = 1.
            y_norm = 1.

        for token in page["tokens"]:
            pos = token["position"]
            offset = token["doc_offset"]
            context.append(
                {
                    'top': pos["top"] * y_norm,
                    'bottom': pos["bottom"] * y_norm,
                    'left': pos["left"] * x_norm,
                    'right': pos["right"] * x_norm,
                    'start': offset["start"],
                    'end': offset["end"],
                    'text': token["text"],
                }
	        )
    return context

def _single_convert_to_finetune(*, document, dpi_norm=True, config={}):
    context = get_context(document, dpi_norm, base_model=config.get('base_model', DocRep))
    texts = []
    offsets = []
    last_end = -1
    num_pages = len(document)
    for i, page in enumerate(document):
        page_obj = page["pages"][0]
        if i == num_pages - 1:
            texts.append(page_obj["text"])
        else:
            texts.append(page_obj["text"] + "\n")
        offset = page_obj["doc_offset"]
        assert offset["start"] == last_end + 1, "If ever this ceases to hold then we have a problem"
        last_end = offset["end"]
    return texts, context


class DocumentPipeline(SequencePipeline):
    def __init__(self, config, multi_label):
        super().__init__(config, multi_label)
        self.config = config

    def text_to_tokens_mask(self, raw_text=None, **kwargs):
        return super().text_to_tokens_mask(**kwargs)

    def zip_list_to_dict(self, X, Y=None, context=None):
        assert context is None
        if Y is not None:
            Y = list(Y)
            if len(X) != len(Y):
                raise FinetuneError("the length of your labels does not match the length of your text")

        out = []
        for i, x in enumerate(X):
            text, context = _single_convert_to_finetune(
                document=x, config=self.config
            )
            joined_text = "".join(text)
            sample = {
                "X": text,
                "raw_text": joined_text,
                # This is done to allow chunk long sequences to rejoin labels for us.
            }
            if self.config.default_context:
                for cii in context:
                    assert cii["text"] == joined_text[cii["start"]: cii["end"]]
                sample["context"] = context
                
            if Y is not None:
                for yii in Y[i]:
                    if "text" in yii:
                        assert yii["text"] == joined_text[yii["start"]: yii["end"]]
                sample["Y"] = Y[i]
            out.append(sample)
        return out
    
    def _text_to_ids(self, X, pad_token=None):
        offset = 0
        for X_page in X:
            for chunk in super()._text_to_ids(X_page, pad_token):
                assert len(chunk.token_starts) == len(chunk.token_ends)
                chunk_dict = chunk._asdict()
                chunk_dict["token_starts"] = [start if start == -1 else start + offset for start in chunk_dict["token_starts"]]
                chunk_dict["token_ends"] = [end if end == -1 else end + offset for end in chunk_dict["token_ends"]]
                chunk_dict["offset"] = offset
                yield EncodedOutput(**chunk_dict)
            offset += len(X_page)

class DocumentLabeler(SequenceLabeler):
    """
    A wrapper to use SequenceLabeler ontop of indico's PDFExtraction APi 
    in ondocument mode with labels at a document charachter level.
    """
    def _get_input_pipeline(self):
        return DocumentPipeline(
            config=self.config, multi_label=self.config.multi_label_sequences
        )

from finetune.target_models.sequence_labeling import SequenceLabeler

def get_context(item, dpi_norm):
    context = []
    for page in item:
        for token in page["tokens"]:
            pos = token["position"]
            offset = token["doc_offset"]
            if dpi_norm:
                dpi = page["pages"][0]["dpi"]
#                dpi = item["pages"][token["page_num"]]["dpi"]
                x_norm = 300 / dpi["dpix"]
                y_norm = 300 / dpi["dpiy"]
            else:
                x_norm = 1.
                y_norm = 1.

            context.append(
                {
                    'top': pos["top"] * y_norm,
		    'bottom': pos["bottom"] * y_norm,
                    'left': pos["left"] * x_norm,
		    'right': pos["right"] * x_norm,
		    'text': token["text"],
		    'start': offset["start"],
		    'end': offset["end"],
                }
	    )
    return context

def reblock_by_offsets(offsets, texts, to_reblock):
    if len(offsets) != len(texts):
        raise ValueError("Length of offsets and texts do not match")
    to_reblock = sorted(to_reblock, key=lambda x: x["start"])
    output = [[] for _ in offsets]
    page_idx = 0
    page_start = offsets[0]["start"]
    page_end = offsets[0]["end"]
    for block in to_reblock:
        block = dict(block)
        while not (page_start <= block["start"] < page_end):
            page_idx += 1
            if page_idx == len(offsets):
                raise ValueError("Block: {} does not not align with offsets")
            page_start = offsets[page_idx]["start"]
            page_end = offsets[page_idx]["end"]
            
        if texts[page_idx][block["start"] - page_start: block["end"] - page_start] != block["text"]:
            raise ValueError("Text does not align for token: {}, is is possible labels span between pages?".format(block))
        block["start"] -= page_start
        block["end"] -=	page_start
        output[page_idx].append(block)
    return output

def _single_convert_to_finetune(document, labels=None, dpi_norm=True, input_level="page", doc_id=None):
    context = get_context(document, dpi_norm)
    texts = []
    offsets = []
    
    if input_level.lower() == "page":
        for page in document:
            page_obj = page["pages"][0]
            texts.append(page_obj["text"])
            offsets.append(page_obj["doc_offset"])
    elif input_level.lower() == "block":
        for page in document:
            for block_obj in page["blocks"]:
                texts.append(block_obj["text"])
                offsets.append(block_obj["doc_offset"])
    else:
        raise ValueError("input_level should be either page or block")
    if doc_id is not None:
        for offset in offsets:
            offset["doc_id"] = doc_id
        
    context_reblocked = reblock_by_offsets(offsets, texts, context)
    if labels is not None:
        labels_reblocked = reblock_by_offsets(offsets, texts, labels)
    else:
        labels_reblocked = None
    return texts, context_reblocked, labels_reblocked, offsets

def finetune_to_odl(preds, offsets, reblock_to_docs=True):
    doc_output = [[] for _ in range(max(offset["doc_id"] for offset in offsets) + 1)]
    output = []
    for pred, offset in zip(preds, offsets):
        pred_out = []
        for pred_i in pred:
            pred_i_copy = dict(pred_i)
            pred_i_copy["start"] += offset["start"]
            pred_i_copy["end"] += offset["start"]
            pred_out.append(pred_i_copy)
        doc_output[offset["doc_id"]].extend(pred_out)
        output.append(pred_out)
        
    if reblock_to_docs:
        return doc_output   
    return output
        
def odl_to_finetune(batch, labels=None, dpi_norm=True, input_level="page"):
    text = []
    context = []
    labels_out = []
    offsets = []
    for doc_id, document in enumerate(batch):
        if labels is not None:
            doc_label = labels[doc_id]
        else:
            doc_label = None
        t, c, l, o = _single_convert_to_finetune(
            document,
            doc_id=doc_id,
            labels=doc_label,
            dpi_norm=dpi_norm,
            input_level=input_level
        )
        text.extend(t)
        context.extend(c)
        labels_out.extend(l)
        offsets.extend(o)
    return text, context, labels_out, offsets
        

class DocumentLabeler(SequenceLabeler):
    def finetune(self, documents, labels):
        text, context, labels_per_page, _ = odl_to_finetune(documents, labels=labels)
        return super().finetune(text, labels_per_page, context=context if self.config.context_dim else None)

    def predict(self, documents):
        text_pred, context_pred, _, offsets_pred = odl_to_finetune(documents, labels=labels_raw)
        preds_raw = model.predict(text_pred, context=context_pred if self.config.context_dim else None)
        return finetune_to_odl(preds_raw, offsets_pred)

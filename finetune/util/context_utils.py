
def get_context_layoutlm(document, dpi_norm):
    context = []
    for page in document:
        for token in page["tokens"]:
            pos = token["position"]
            offset = token["doc_offset"]
            width = page["pages"][0]["size"]["width"]
            height = page["pages"][0]["size"]["height"]
            context.append(
                {
                    "top": int(pos["top"] / height * 1000),
                    "bottom": int(pos["bottom"] / height * 1000),
                    "left": int(pos["left"] / width * 1000),
                    "right": int(pos["right"] / width * 1000),
                    "start": offset["start"],
                    "end": offset["end"],
                    "text": token["text"],
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
            x_norm = 1.0
            y_norm = 1.0

        for token in page["tokens"]:
            pos = token["position"]
            offset = token["doc_offset"]
            context.append(
                {
                    "top": pos["top"] * y_norm,
                    "bottom": pos["bottom"] * y_norm,
                    "left": pos["left"] * x_norm,
                    "right": pos["right"] * x_norm,
                    "start": offset["start"],
                    "end": offset["end"],
                    "text": token["text"],
                }
            )
    return context

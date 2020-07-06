import unicodedata


WEIRD_SPM_CHAR = "▁"

def normalize_nfkc(text):
    chars = [unicodedata.normalize('NFKC', t) for t in text]
    lookup = []
    text_out = ""
    for i, c in enumerate(text):
        normed = unicodedata.normalize('NFKC', c)
        lookup += [i for _ in normed]
        text_out += normed
    lookup.append(len(text))
    return lookup, text_out

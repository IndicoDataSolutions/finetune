import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

from finetune import Association
from finetune.datasets import Dataset, generic_download

logging.basicConfig(level=logging.DEBUG)

from nltk.corpus import treebank

idx = 0

def find_pairs_from_vp(vp_tree):
    global idx
    idx_local = idx
    idx += 1
    out = []
    for branch in vp_tree:
        for l in ["VB", "N"]:
            if branch.label().startswith(l):
                out.append(((l, idx), " ".join(branch.flatten())))
                break
        else:
            out.extend(find_noun_verb_pairs(branch))
    return out


def find_noun_verb_pairs(tree):
    if type(tree) == str:
        return [(None, tree)]

    if tree.label() == "-NONE-":
        return []

    if tree.label() == "VP":
        return find_pairs_from_vp(tree)
    else:
        out = []
        for branch in tree:
            out.extend(find_noun_verb_pairs(branch))
    return out


def clean(tagged_sentence):
    tags, word = list(zip(*tagged_sentence))
    nouns = [t[1] if t and t[0] == "N" else None for t in tags]
    verbs = [t[1] if t and t[0] == "VB" and t[1] in nouns else None for t in tags]
    nouns = [nt if nt in verbs else None for nt in nouns]

    nouns_contract = []
    verbs_contract = []
    words_contract = []

    for n, v, w in zip(nouns, verbs, word):
        if not nouns_contract or v != verbs_contract[-1] or n != nouns_contract[-1]:
            nouns_contract.append(n)
            verbs_contract.append(v)
            words_contract.append(w)
        else:
            words_contract[-1] += ' ' + w
    final_segment_verbs = [v for v in verbs_contract if v is not None]
    associations = [final_segment_verbs.index(idx) if idx else None for idx in nouns_contract]
    output_lab = []
    output_text = ""
    for n, v, w, a in zip(nouns_contract, verbs_contract, words_contract, associations):
        if n or v:
            offset = 1 if output_text else 0
            output_lab.append(
                {
                    "start": len(output_text) + offset,
                    "end": len(output_text) + len(w) + offset,
                    "text": w,
                    "label": "noun_phrase" if n else "verb",
                }
            )
            if a is not None:
                output_lab[-1]["association"] = {
                    "index": a,
                    "relationship": "has_verb"
                }
        if output_text:
            output_text += ' '
        output_text += w

    return output_text, output_lab

if __name__ == "__main__":
    # Train and evaluate on SST
    X = []
    Y = []
    for i, a in enumerate(treebank.parsed_sents()):
        x, y = clean(find_noun_verb_pairs(a))
        X.append(x)
        Y.append(y)
    model = Association(possible_associations=["has_verb"], max_length=32)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=42)
    model.fit(trainX, trainY)
    print(model.predict(testX))

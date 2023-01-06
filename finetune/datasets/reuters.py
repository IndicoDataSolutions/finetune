import os
import requests
import codecs
import json
import hashlib
import io
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from sklearn.model_selection import train_test_split

from finetune import SequenceLabeler
from finetune.datasets import Dataset
from finetune.base_models import GPT, GPT2, TCN, RoBERTa
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.util.metrics import annotation_report, sequence_labeling_token_confusion

XML_PATH = os.path.join("Data", "Sequence", "reuters.xml")
DATA_PATH = os.path.join("Data", "Sequence", "reuters.json")
CHECKSUM = "a79cab99ed30b7932d46711ef8d662e0"

from colorama import Back

def overlaps(a, b):
    return a["start"] < b["end"] <= a["end"] or b["start"] < a["end"] <= b["end"]

def get_preds_labels_comparison(text, labels, preds):
    char_labels = [None for _ in text]
    char_preds = [None for _ in text]
    for l in labels:
        for i in range(l["start"], l["end"]):
            char_labels[i] = l["label"]
    for p in preds:
        for i in range(p["start"], p["end"]):
            char_preds[i] = p["label"]

    output = []
    for i, (p, l) in enumerate(zip(char_preds, char_labels)):
        if p is not None or l is not None:
            if (
                    output
                    and output[-1]["end"] == i
                    and output[-1]["pred"] == p
                    and output[-1]["label"] == l
                    and text[i: i + 1].strip("\n")
            ):
                output[-1]["end"] = i + 1
            else:
                output.append(
                    {
                        "start": i,
                        "end": i + 1,
                        "label": l,
                        "pred": p,
                    }
                )
    return output

def remove_overlaps(labels):
    labels_out = []
    for l in labels:
        for lo in labels_out:
            if overlaps(lo, l):
                break
        else:
            labels_out.append(l)
    return labels_out

def adjust_positions(after, labels, offset=1):
    for l in labels:
        if l["start"] >= after:
            l["start"] += offset
        if l["end"] > after:
            l["end"] += offset
    return labels

def get_color_text(text, labels, preds, chunk_bounds):
    labels = remove_overlaps(labels)
    comparison = get_preds_labels_comparison(text, labels, preds)
    original_text = text
    colours = [Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN]
    for cb in sorted(chunk_bounds, key=lambda x: -x["start"]):
        join_txt = "✂️"
        text = text[: cb["start"]] + join_txt + text[cb["start"]:]
        comparison = adjust_positions(cb["start"], comparison, offset=len(join_txt))
    comparison = sorted(comparison, key=lambda x:-x["start"])
    for c in comparison:
        col = Back.GREEN if c["label"] == c["pred"] else Back.RED
        text = text[:c["start"]] + col+ text[c["start"]: c["end"]] + Back.RESET + text[c["end"]:]
    return text


class Reuters(Dataset):

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    @property
    def md5(self):
        return CHECKSUM

    def download(self):

        url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
        r = requests.get(url)

        with open(XML_PATH, 'wb') as fd:
            fd.write(r.content)

        fd = open(XML_PATH)
        soup = bs(fd, "html.parser")
        docs = []
        docs_labels = []
        for elem in soup.find_all("document"):
            texts = []
            labels = []

            # Loop through each child of the element under "textwithnamedentities"
            for c in elem.find("textwithnamedentities").children:
                if type(c) == Tag:
                    if c.name == "namedentityintext":
                        label = "Named Entity"  # part of a named entity
                    else:
                        label = "<PAD>"  # irrelevant word
                    texts.append(c.text)
                    labels.append(label)

            docs.append(texts)
            docs_labels.append(labels)

        fd.close()
        os.remove(XML_PATH)

        raw_texts = ["".join(doc) for doc in docs]
        texts, annotations = finetune_to_indico_sequence(raw_texts, docs, docs_labels, none_value="<PAD>",
                                                         subtoken_predictions=True)
        df = pd.DataFrame({'texts': texts, 'annotations': [json.dumps(annotation) for annotation in annotations]})
        df.to_csv(DATA_PATH)


if __name__ == "__main__":
    dataset = Reuters().dataframe
    dataset['annotations'] = [json.loads(annotation) for annotation in dataset['annotations']]
    trainX, testX, trainY, testY = train_test_split(
        dataset.texts.values,
        dataset.annotations.values,
        test_size=0.2,
        random_state=42
    )
    model = SequenceLabeler(batch_size=8, n_epochs=1, val_size=0.0, max_length=32, chunk_long_sequences=True, subtoken_predictions=False, crf_sequence_labeling=True, multi_label_sequences=False, min_steps=2048, predict_chunk_markers=True)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    for t, l, p in zip(testX, testY, predictions):
        print(get_color_text(t, l, p["prediction"], p["chunks"]))
        input(">>>>")


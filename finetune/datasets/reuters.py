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

from finetune.datasets import Dataset
from finetune import SequenceLabeler
from finetune.utils import finetune_to_indico_sequence
from finetune.metrics import annotation_report

XML_PATH = os.path.join("Data", "Sequence", "reuters.xml")
DATA_PATH = os.path.join("Data", "Sequence", "reuters.json")
CHECKSUM = "a79cab99ed30b7932d46711ef8d662e0"

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
        soup = bs(fd, "html5lib")
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
        texts, annotations = finetune_to_indico_sequence(raw_texts, docs, docs_labels)
        df = pd.DataFrame({'texts': texts, 'annotations': [json.dumps(annotation) for annotation in annotations]})
        df.to_csv(DATA_PATH)


if __name__ == "__main__":
    dataset = Reuters().dataframe
    dataset['annotations'] = [json.loads(annotation) for annotation in dataset['annotations']]
    trainX, testX, trainY, testY = train_test_split(
        dataset.texts.values, 
        dataset.annotations.values, 
        test_size=0.3,
        random_state=42
    )
    model = SequenceLabeler(batch_size=2, val_size=0., chunk_long_sequences=True)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    print(annotation_report(testY, predictions))

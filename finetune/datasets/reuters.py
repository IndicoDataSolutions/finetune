import os
import requests
import codecs
import json
import io

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from sklearn.model_selection import train_test_split

from finetune.datasets import Dataset
from finetune import SequenceLabeler
from finetune.utils import finetune_to_indico_sequence

DATA_PATH = os.path.join("Data", "Sequence", "reuters.json")


class Reuters(Dataset):


    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    def download(self):

        if os.path.exists(self.filename):
            return
           
        url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
        r = requests.get(url)
        fd = io.StringIO()
        fd.write(r.content)
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

        texts, annotations = finetune_to_indico_sequence(docs, docs_labels)
        df = pd.DataFrame({'texts': texts, 'annotations': [json.dumps(annotation) for annotation in annotations]})
        df.to_csv(Reuters.DATA_PATH)


if __name__ == "__main__":
    dataset = Reuters(nrows=1500).dataframe
    dataset['annotations'] = [json.loads(annotation) for annotation in dataset['annotations']]
    trainX, testX, trainY, testY = train_test_split(dataset.text, dataset.annotations, test_size=0.3, random_state=42)
    model = SequenceLabeler(verbose=False)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)

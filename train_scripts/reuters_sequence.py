import os
import requests
from finetune import LanguageModelSequence

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import json
from sklearn.model_selection import train_test_split

dataset_path = os.path.join("reuters.xml")
processed_path = os.path.join('reuters.json')

if not os.path.exists(dataset_path):
    url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
    r = requests.get(url)
    with open(dataset_path, "wb") as fp:
        fp.write(r.content)

    with codecs.open(dataset_path, "r", "utf-8") as infile:
        soup = bs(infile, "html5lib")

    docs = []
    docs_no_lab = []
    docs_just_lab = []
    for elem in soup.find_all("document"):
        texts = []
        texts_no_labels = []
        just_labels = []

        # Loop through each child of the element under "textwithnamedentities"
        for c in elem.find("textwithnamedentities").children:
            if type(c) == Tag:
                if c.name == "namedentityintext":
                    label = "N"  # part of a named entity
                else:
                    label = "I"  # irrelevant word
                texts.append([c.text, label])
                texts_no_labels.append(c.text)
                just_labels.append(label)

        docs_no_lab.append("".join(texts_no_labels))
        docs_just_lab.append(just_labels)
        docs.append(texts)
    with open(processed_path, 'wt') as fp:
        json.dump((docs, docs_no_lab, docs_just_lab), fp)


with open(processed_path, 'rt') as fp:
    dataset, dataset_no_labels, just_labels = json.load(fp)

dataset_train, _, _, dataset_test = train_test_split(dataset, dataset_no_labels, test_size=0.3)

save_file_autosave = 'tests/saved-models/autosave_path'

model = LanguageModelSequence(verbose=False, autosave_path=save_file_autosave)
model.fit(list(zip(*[dataset_train])))
predictions = model.predict(list(zip(*[dataset_test])))
model.save(save_file_autosave)
print(predictions)
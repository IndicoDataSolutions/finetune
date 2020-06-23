import os
import json
import unittest

import pandas as pd

from sklearn.model_selection import train_test_split

from finetune import DocumentLabeler
from finetune.base_models import DocRep

from finetune.util.metrics import sequence_labeling_micro_token_f1
    



DATA_PATH = os.path.join("Data", "Sequence", "doc_rep_integration.csv")

class TestDocumentLabeler(unittest.TestCase):
    def setUp(self):
        self.labels = [
            [
                {'start': 15, 'end': 27, 'label': 'dodge county', 'text': 'Dodge County'},
                {'start': 2836, 'end': 2848, 'label': 'dodge county', 'text': 'Dodge County'},
                {'start': 2981, 'end': 2993, 'label': 'dodge county', 'text': 'Dodge County'},
                {'start': 3054, 'end': 3066, 'label': 'dodge county', 'text': 'Dodge County'},
                {'start': 3204, 'end': 3216, 'label': 'dodge county', 'text': 'Dodge County'}
            ],
            [
                {'start': 4, 'end': 21, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 251, 'end': 268, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 359, 'end': 376, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 981, 'end': 998, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 1296, 'end': 1313, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 2023, 'end': 2040, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 2450, 'end': 2467, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 2963, 'end': 2980, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 4713, 'end': 4730, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 5445, 'end': 5462, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 6642, 'end': 6659, 'label': 'city of hollywood', 'text': 'City of Hollywood'},
                {'start': 7082, 'end': 7099, 'label': 'city of hollywood', 'text': 'City of Hollywood'}
            ]
        ]
        with open("tests/data/test_ocr_documents.json", "rt") as fp:
            self.documents = json.load(fp)

    def test_fit_predict(self):
        model = DocumentLabeler(n_epochs=20)
        model.fit(self.documents, self.labels)
        preds = model.predict(self.documents)
        self.assertEqual(len(preds), len(self.documents))
        for pred, lab in zip(preds, self.labels):
            # checks that an overfit model, will produce the exact same output as was given as
            # input even after being sliced up and put back together.
            for p, l in zip(pred, lab):
                del p["confidence"]
                self.assertEqual(p, l)

    def test_fit_predict_doc_rep(self):
        model = DocumentLabeler(n_epochs=20, base_model=DocRep, crf_sequence_labeling=True)
        model.fit(self.documents, self.labels)
        preds = model.predict(self.documents)
        self.assertEqual(len(preds), len(self.documents))
        for pred, lab in zip(preds, self.labels):
            # checks that an overfit model, will produce the exact same output as was given as                                                        
            # input even after being sliced up and put back together.
            print(pred, lab)
            for p, l in zip(pred, lab):
                del p["confidence"]
                self.assertEqual(p, l)

    def test_integration(self):
        df = pd.read_csv(DATA_PATH)
        
        ocr = [json.loads(o) for o in df.ocr.values]
        labels = [json.loads(l) for l in df.labels.values]

        train_ocr, test_ocr, train_labels, test_labels = train_test_split(ocr, labels, random_state=42, test_size=0.2)

        model = DocumentLabeler(base_model=DocRep)
        model.fit(train_ocr, train_labels)
        model_preds = model.predict(test_ocr)
        model_f1 = sequence_labeling_micro_token_f1(test_labels, model_preds)

        baseline_model = DocumentLabeler() # RoBERTa
        baseline_model.fit(train_ocr, train_labels)
        baseline_model_preds = baseline_model.predict(test_ocr)
        baseline_model_f1 = sequence_labeling_micro_token_f1(test_labels, baseline_model_preds)
        self.assertGreater(model_f1, baseline_model_f1)
        self.assertGreater(model_f1, 0.95)

    def test_featurize_doc_with_empty_pages(self):
        """ Test that context alignment with tokenization doesn't error out """
        with open('tests/data/doc_with_empty_pages.json') as f:
            doc = json.load(f)
        model = DocumentLabeler(n_epochs=20, base_model=DocRep, crf_sequence_labeling=True)
        model.featurize([doc])
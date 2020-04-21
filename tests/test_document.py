import json
import unittest
from finetune import DocumentLabeler
from finetune.base_models import DocRep

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
            for p, l in zip(pred, lab):
                del p["confidence"]
                self.assertEqual(p, l)


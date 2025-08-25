import json
import os

import pandas as pd
import pytest
from sequence_metrics.metrics import sequence_labeling_micro_token_f1
from sklearn.model_selection import train_test_split

from finetune import DocumentLabeler
from finetune.base_models import DocRep, RoBERTa, TestingModel

DATA_PATH = os.path.join("tests", "data", "doc_rep_integration.csv")


@pytest.fixture
def ocr_documents_and_labels():
    labels = [
        [
            {"start": 15, "end": 27, "label": "dodge county", "text": "Dodge County"},
            {"start": 2836, "end": 2848, "label": "dodge county", "text": "Dodge County"},
            {"start": 2981, "end": 2993, "label": "dodge county", "text": "Dodge County"},
            {"start": 3054, "end": 3066, "label": "dodge county", "text": "Dodge County"},
            {"start": 3204, "end": 3216, "label": "dodge county", "text": "Dodge County"},
        ],
        [
            {"start": 4, "end": 21, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 251, "end": 268, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 359, "end": 376, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 981, "end": 998, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 1296, "end": 1313, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 2023, "end": 2040, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 2450, "end": 2467, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 2963, "end": 2980, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 4713, "end": 4730, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 5445, "end": 5462, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 6642, "end": 6659, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 7082, "end": 7099, "label": "city of hollywood", "text": "City of Hollywood"},
        ],
    ]
    with open("tests/data/test_ocr_documents.json", "rt") as fp:
        documents = json.load(fp)
    return documents, labels


def test_fit_predict(ocr_documents_and_labels, get_untrained_document_labeler):
    documents, labels = ocr_documents_and_labels
    model = get_untrained_document_labeler(n_epochs=1, base_model=TestingModel)
    model.fit(documents, labels)
    preds = model.predict(documents)
    assert len(preds) == len(documents)
    for pred, lab in zip(preds, labels):
        for p, l in zip(pred, lab):
            del p["confidence"]
            assert p == l


def test_fit_predict_doc_rep_1_block(ocr_documents_and_labels, get_untrained_document_labeler):
    documents, labels = ocr_documents_and_labels
    model = get_untrained_document_labeler(n_layer=1, n_epochs=1, batch_size=1, base_model=DocRep, crf_sequence_labeling=True)
    model.fit(documents, labels)
    preds = model.predict(documents)
    assert len(preds) == len(documents)
    for pred, lab in zip(preds, labels):
        for p, l in zip(pred, lab):
            del p["confidence"]
            assert p == l

@pytest.mark.skip(reason="Skipping integration test. Expensive and not currently critical path.")
def test_integration():
    df = pd.read_csv(DATA_PATH)

    ocr = [json.loads(o) for o in df.ocr.values]
    labels = [json.loads(l) for l in df.labels.values]

    train_ocr, test_ocr, train_labels, test_labels = train_test_split(
        ocr, labels, random_state=42, test_size=0.2
    )

    model = DocumentLabeler(base_model=DocRep)
    model.fit(train_ocr, train_labels)
    model_preds = model.predict(test_ocr)
    model_f1 = sequence_labeling_micro_token_f1(test_labels, model_preds)
    assert model_f1 > 0.95


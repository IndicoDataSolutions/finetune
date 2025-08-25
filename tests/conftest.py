import shutil
from pathlib import Path

import tensorflow as tf
import pandas as pd
import pytest
import codecs
import json
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import requests
import os

from finetune import Classifier, SequenceLabeler, DocumentLabeler, MultiLabelClassifier
from finetune.base_models import TestingModel, TextCNN, TCNModel, RoBERTa, ModernBertModel, BERTModelCased
from finetune.datasets import generic_download
from finetune.base_models import TestingModel
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.nn.target_blocks import Classifier as ClassifierBlock, MultiClassifier as MultiLabelClassifierBlock, SequenceLabeler as SequenceLabelerBlock
import finetune.base

@pytest.fixture(scope="function")
def save_model_dir():
    path = Path("tests") / "saved-models"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path)

@pytest.fixture(scope="module")
def sst_dataset():
    path = Path("tests") / "data" / "SST-binary.csv"
    yield pd.read_csv(path, nrows=60)

@pytest.fixture(scope="module")
def sst_dataset():
    path = Path("tests") / "data" / "SST-binary.csv"
    yield pd.read_csv(path, nrows=60)

@pytest.fixture(scope="module")
def reuters_dataset():
    path = Path("Data") / "Sequence" / "reuters.xml"
    processed_path = Path("Data") / "Sequence" / "reuters.json"
    if not processed_path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path):
            url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
            r = requests.get(url)
            with open(path, "wb") as fp:
                fp.write(r.content)

        with codecs.open(path, "r", "utf-8") as infile:
            soup = bs(infile, "html.parser")

        docs = []
        docs_labels = []
        for elem in soup.find_all("document"):
            texts = []
            labels = []
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
        with open(processed_path, "wt") as fp:
            json.dump((docs, docs_labels), fp)
    with open(processed_path, "rt") as fp:
        docs, docs_labels = json.load(fp)
    yield docs, docs_labels

@pytest.fixture(scope="module")
def get_untrained_classifier():
    config = {"batch_size": 2, "max_length": 128, "n_epochs": 1, "base_model": TestingModel}
    def _get_model(**config_overrides):
        config.update(config_overrides)
        return Classifier(**config)
    return _get_model

@pytest.fixture(scope="module")
def get_untrained_sequence_labeler():
    base_config = {"batch_size": 2, "max_length": 256, "n_epochs": 2, "base_model": TestingModel}
    def _get_model(**config_overrides):
        cfg = {**base_config, **config_overrides}
        return SequenceLabeler(**cfg)
    return _get_model

@pytest.fixture(scope="module")
def get_untrained_document_labeler():
    base_config = {"n_epochs": 2, "base_model": TestingModel}
    def _get_model(**config_overrides):
        cfg = {**base_config, **config_overrides}
        return DocumentLabeler(**cfg)
    return _get_model

@pytest.fixture(scope="module")
def get_untrained_multilabel_classifier():
    base_config = {"batch_size": 2, "max_length": 128, "n_epochs": 1}
    def _get_model(**config_overrides):
        cfg = {**base_config, **config_overrides}
        return MultiLabelClassifier(**cfg)
    return _get_model

@pytest.fixture(scope="module")
def trained_classifier(sst_dataset, get_untrained_classifier):
    model = get_untrained_classifier()
    train_sample = sst_dataset.sample(n=20)
    model.fit(train_sample.Text.values, train_sample.Target.values)
    yield model
    model.close()

@pytest.fixture(scope="module")
def classification_text_sample(sst_dataset):
    def fn(n=20):
        return sst_dataset.sample(n=n)
    return fn

@pytest.fixture(params=[TextCNN, TCNModel, RoBERTa, ModernBertModel, BERTModelCased])
def base_model(request):
    return request.param

@pytest.fixture(scope="module")
def reuters_indico_sequence(reuters_dataset):
    docs, docs_labels = reuters_dataset
    raw_docs = ["".join(text) for text in docs]
    temp_model = SequenceLabeler()
    texts, annotations = finetune_to_indico_sequence(
        raw_docs, docs, docs_labels, none_value=temp_model.config.pad_token
    )
    return texts, annotations

# ---- Reusable tiny datasets / pretrained models ----

@pytest.fixture(scope="module")
def tiny_classification_corpus():
    X = ["cat", "kitten", "purr", "finance", "stocks", "bonds"]
    y = ["cat", "cat", "cat", "finance", "finance", "finance"]
    return X, y

@pytest.fixture(scope="module")
def pretrained_classifier_testingmodel(tiny_classification_corpus, save_model_dir):
    X, y = tiny_classification_corpus
    model = Classifier(base_model=TestingModel, n_epochs=1, batch_size=2)
    model.fit(X, y)
    model_path = save_model_dir / "testingmodel_classifier.jl"
    model.save(model_path)
    yield model, str(model_path)
    model.close()

@pytest.fixture(scope="module")
def small_sequence_dataset():
    texts = [
        "John lives in New York",
        "Mary works at OpenAI",
    ]
    labels = [
        [
            {"start": 0, "end": 4, "label": "PERSON", "text": "John"},
            {"start": 14, "end": 22, "label": "LOCATION", "text": "New York"},
        ],
        [
            {"start": 0, "end": 4, "label": "PERSON", "text": "Mary"},
            {"start": 14, "end": 20, "label": "ORG", "text": "OpenAI"},
        ],
    ]
    return texts, labels

@pytest.fixture(scope="module")
def pretrained_sequence(small_sequence_dataset):
    texts, labels = small_sequence_dataset
    model = SequenceLabeler(base_model=TestingModel, n_epochs=1, batch_size=2)
    model.fit(texts, labels)
    yield model
    model.close()

@pytest.fixture(scope="module")
def small_ocr_docs():
    docs = [
        [{"pages": [{"text": "City of Hollywood in Dodge County"}]}],
        [{"pages": [{"text": "Dodge County and City of Hollywood"}]}],
    ]
    labels = [
        [
            {"start": 0, "end": 16, "label": "city of hollywood", "text": "City of Hollywood"},
            {"start": 20, "end": 32, "label": "dodge county", "text": "Dodge County"},
        ],
        [
            {"start": 0, "end": 12, "label": "dodge county", "text": "Dodge County"},
            {"start": 17, "end": 33, "label": "city of hollywood", "text": "City of Hollywood"},
        ],
    ]
    return docs, labels

@pytest.fixture(scope="module")
def pretrained_document_labeler(small_ocr_docs):
    docs, labels = small_ocr_docs
    model = DocumentLabeler(base_model=TestingModel, n_epochs=1)
    model.fit(docs, labels)
    yield model
    model.close()


@pytest.fixture
def mock_get_keras_model(monkeypatch):
    models = []
    def mock_get_keras_model(
        target_block,
        encoder,
        target_dim,
        label_encoder,
        config,
        train_input_signature,
        predict_input_signature,
        use_xla,
        **model_kwargs
    ):
        class MockModel:
            def __init__(self, *args, **kwargs):
                self.target_block = target_block
                self.encoder = encoder
                self.target_dim = target_dim
                self.label_encoder = label_encoder
                self.config = config
                self.train_input_signature = train_input_signature
                self.predict_input_signature = predict_input_signature
                self.use_xla = use_xla
                self.model_kwargs = model_kwargs
                self.fit_calls = []
                self.predict_calls = []
                self.compile_calls = []

                # For Mocking the attributes expected to exist on the keras model
                self._build = False
                self.name = "finetune_model"
                self._layers = []
                self.weights = []
                self.variables = []

            def compile(self, *args, **kwargs):
                self.compile_calls.append({
                    "args": args,
                    "kwargs": kwargs,
                })
                return

            def finetune_predict(self, data, *args, **kwargs):
                self.predict_calls.append({
                    "data": data,
                    "args": args,
                    "kwargs": kwargs,
                })
                tokens = data["tokens"]
                batch_size = tokens.shape[0]
                sequence_length = tokens.shape[1]

                if isinstance(self.target_block, ClassifierBlock):
                    preds = tf.zeros(shape=(batch_size), dtype=tf.int32)
                    probas = tf.nn.softmax(tf.random.normal(shape=(batch_size, self.target_dim)), axis=-1)
                elif isinstance(self.target_block, MultiLabelClassifierBlock):
                    preds = tf.zeros(shape=(batch_size, self.target_dim), dtype=tf.int32)
                    probas = tf.nn.sigmoid(tf.random.normal(shape=(batch_size, self.target_dim)), axis=-1)
                elif isinstance(self.target_block, SequenceLabelerBlock):
                    probas = tf.nn.softmax(tf.random.normal(shape=(batch_size, sequence_length, self.target_dim)), axis=-1)
                    preds = tf.argmax(probas, axis=-1)
                else:
                    raise ValueError(f"Unknown target block type for mock: {type(self.target_block)}")

                return {
                    "sequence_features": tf.random.normal(shape=(batch_size, sequence_length, self.config.n_embed)),
                    "features": tf.random.normal(shape=(batch_size, self.config.n_embed)),
                    "preds": preds,
                    "probas": probas,
                }

            def __call__(self, *args, **kwargs):
                self._build = True

            def fit(self, data, *args, **kwargs):
                self.fit_calls.append({
                    "data": data,
                    "args": args,
                    "kwargs": kwargs,
                })
                return

        model = MockModel()
        models.append(model)
        return model
    with monkeypatch.context() as m:
        m.setattr(finetune.base, "get_keras_model", mock_get_keras_model)
        yield models
    

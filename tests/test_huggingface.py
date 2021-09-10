import os
import json

import unittest
import numpy as np
import tensorflow as tf
import joblib as jl

from transformers import AutoTokenizer, TFAutoModel, BertTokenizer
from finetune import SequenceLabeler, DocumentLabeler
from finetune.base_models import LayoutLM
from finetune.base_models.huggingface.models import (
    HFBert,
    HFElectraGen,
    HFElectraDiscrim,
    HFXLMRoberta,
    HFT5,
    HFAlbert,
    HFLongformer,
)
from finetune.target_models.seq2seq import HFS2S
from sklearn.model_selection import train_test_split
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence

class TestHuggingFace(unittest.TestCase):
    def setUp(self):
        with open('tests/data/weird_text.txt') as f:
            weird_text = ''.join(f.readlines())
        # This token is added to our t5 model, causes differences if it appears in the text
        self.text = weird_text[:1000].replace("<", "")

    def check_embeddings_equal(self, finetune_base_model, hf_model_path, decimal=2, use_fast=True, **kwargs):
        finetune_model = SequenceLabeler(
            base_model=finetune_base_model,
            chunk_long_sequences=False,
        )
        finetune_seq_features = finetune_model.featurize_sequence([self.text])[0]
        hf_seq_features = self.huggingface_embedding(self.text, hf_model_path, use_fast=use_fast, **kwargs)[0]
        if len(finetune_seq_features) + 2 == len(hf_seq_features):
            hf_seq_features = hf_seq_features[1:-1]
        if len(finetune_seq_features) + 1 == len(hf_seq_features):
            hf_seq_features = hf_seq_features[:-1]
        np.testing.assert_array_almost_equal(
            finetune_seq_features, hf_seq_features, decimal=decimal,
        )
        finetune_model.fit([self.text], [[{"start": 0, "end": 4, "label": "class_a"}]])

    def huggingface_embedding(self, text, model_path, use_fast=True):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        model = TFAutoModel.from_pretrained(model_path)
        tokens = tokenizer.encode(self.text)
        input_ids = tf.constant(tokens)[None, :]  # Batch size 1
        kwargs = {
            "inputs_embeds": None,
            "training": False,
        }
        if model.config.is_encoder_decoder:
            outputs = model.encoder(input_ids, **kwargs)
        else:
            outputs = model(
                input_ids, **kwargs
            )  # outputs is tuple where first element is sequence features
        last_hidden_states = outputs[0]
        return last_hidden_states.numpy()

    def test_xlm_roberta(self):
        self.check_embeddings_equal(HFXLMRoberta, "jplu/tf-xlm-roberta-base")

    def test_bert(self):
        self.check_embeddings_equal(HFBert, "bert-base-uncased")

    def test_t5_s2s(self):
        text = "sequence test { text }"
        finetune_model = HFS2S(
            base_model=HFT5,
            n_epochs=100,
            batch_size=1,
        )
        finetune_model.fit([text] * 5, [text] * 5)
        finetune_model.save("test.jl")
        self.assertEqual(finetune_model.predict([text]), [text])
        self.assertEqual(len(finetune_model.predict([text, "Some other text"])), 2)
        loaded_model = HFS2S.load("test.jl")
        self.assertEqual(loaded_model.predict([text]), [text])

    def test_t5_s2s_ner(self):
        with open(os.path.join('Data', 'Sequence', 'reuters.json'), "rt") as fp:
            texts, labels = json.load(fp)
        raw_docs = ["".join(text) for text in texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, texts, labels, none_value="<PAD>")
        labels = [" | ".join(x["text"] for x in a) for a in annotations]
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, labels, test_size=0.1, random_state=42
        )
        finetune_model = HFS2S(
            base_model=HFT5,
        )
        finetune_model.fit(train_texts, train_annotations)
        seps_included = False
        pred_correct = False
        for t, p, l in zip(test_texts, finetune_model.predict(test_texts), test_annotations):
            seps_included = seps_included or " | " in p # check it predicts separators
            pred_correct = pred_correct or any(li in p for li in l.split(" | ")) # at least one extraction is predicted correctly.

        self.assertTrue(seps_included)
        self.assertTrue(pred_correct)

    def test_t5(self):
        # Huggingface fast tokenizer is a bit broken and does weird things with whitespace.
        self.check_embeddings_equal(HFT5, "t5-base", use_fast=False, decimal=1)

    def test_albert(self):
        self.check_embeddings_equal(HFAlbert, "albert-base-v2")

    def test_longformer(self):
        self.check_embeddings_equal(HFLongformer, "allenai/longformer-base-4096")

    def test_layoutlm(self):
        activations_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "test-layoutlm-activations.jl")
        def subset_doc(doc):
            """ Take approximately the first 512 tokens from just the first page """
            for page in doc:
                page["tokens"] = page["tokens"][:60]
                page_idx_end = page["tokens"][-1]["page_offset"]["end"]
                page["pages"][0]["text"] = page["pages"][0]["text"][:page_idx_end]
                page["pages"][0]["doc_offset"]["end"] = page["pages"][0]["doc_offset"]["start"] + page_idx_end
                return [page]

        with open("tests/data/test_ocr_documents.json", "rt") as fp:
            documents = json.load(fp)
        documents = [subset_doc(doc) for doc in documents]
        finetune_model = DocumentLabeler(base_model=LayoutLM)
        expected_all = jl.load(activations_path)
        for doc, expected in zip(documents, expected_all):
            finetune_seq_features = finetune_model.featurize_sequence([doc])
            np.testing.assert_array_almost_equal(
                finetune_seq_features, expected, decimal=1,
            )

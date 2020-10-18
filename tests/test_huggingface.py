import os
import json

import unittest
import numpy as np
import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModel
from finetune import SequenceLabeler
from finetune.base_models.huggingface.models import (
    HFBert,
    HFElectraGen,
    HFElectraDiscrim,
    HFXLMRoberta,
    HFT5,
    HFAlbert,
)
from finetune.datasets.reuters import Reuters
from finetune.target_models.seq2seq import HFS2S
from sklearn.model_selection import train_test_split
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence


class TestHuggingFace(unittest.TestCase):
    def setUp(self):
        with open('tests/data/weird_text.txt') as f:
            weird_text = ''.join(f.readlines())
        self.text = weird_text[:1000]

    def check_embeddings_equal(self, finetune_base_model, hf_model_path):
        finetune_model = SequenceLabeler(
            base_model=finetune_base_model,
            train_embeddings=False,
            n_epochs=1,
            batch_size=1,
        )
        finetune_seq_features = finetune_model.featurize_sequence([self.text])[0]
        hf_seq_features = self.huggingface_embedding(self.text, hf_model_path)[0]
        if len(finetune_seq_features) + 2 == len(hf_seq_features):
            hf_seq_features = hf_seq_features[1:-1]
        np.testing.assert_array_almost_equal(
            finetune_seq_features, hf_seq_features, decimal=5,
        )
        finetune_model.fit([self.text], [[{"start": 0, "end": 4, "label": "class_a"}]])

    def huggingface_embedding(self, text, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModel.from_pretrained(model_path)
        input_ids = tf.constant(tokenizer.encode(self.text))[None, :]  # Batch size 1

        if model.config.is_encoder_decoder:
            # Need to decide how to properly handle decoder input ids
            # This ain't it
            outputs = model.encoder(input_ids)
        else:
            outputs = model(
                input_ids,
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
            n_epochs=30,
            batch_size=2,
        )
        finetune_model.fit([text] * 5, [text] * 5)
        finetune_model.save("test.jl")
        self.assertEqual(finetune_model.predict([text]), [text])
        self.assertEqual(len(finetune_model.predict([text, "Some other text"])), 2)
        loaded_model = HFS2S.load("test.jl")
        self.assertEqual(loaded_model.predict([text]), [text])

    def test_t5_s2s_ner(self):
        df = Reuters().dataframe
        texts = df.texts.values
        labels = [json.loads(entry) for entry in df.annotations]
        labels = [" | ".join(x["text"] for x in a) for a in labels]
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, labels, test_size=0.1, random_state=42
        )
        finetune_model = HFS2S(
            base_model=HFT5,
            n_epochs=3,
            batch_size=16,
            low_memory_mode=True, 
        )
        finetune_model.fit(train_texts, train_annotations)
        seps_included = False
        pred_correct = False
        for t, p, l in zip(test_texts, finetune_model.predict(test_texts), test_annotations):
            seps_included = seps_included or " | " in p # check it predicts separators
            pred_correct = pred_correct or any(li in p for li in l.split(" | ")) # at least one extraction is predicted correctly.

        self.assertTrue(seps_included)2
        self.assertTrue(pred_correct)
    
    def test_t5_s2s_ner_label_smoothing(self):
        df = Reuters().dataframe
        texts = df.texts.values
        labels = [json.loads(entry) for entry in df.annotations]
        labels = [" | ".join(x["text"] for x in a) for a in labels]
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, labels, test_size=0.1, random_state=42
        )
        finetune_model = HFS2S(
            base_model=HFT5,
            n_epochs=3,
            batch_size=16,
            # s2s_label_smoothing=0.1,
            # s2s_smoothing_mean_targets=False
        )
        print(train_texts, train_annotations)
        finetune_model.fit(train_texts, train_annotations)
        seps_included = False
        pred_correct = False
        for t, p, l in zip(test_texts, finetune_model.predict(test_texts), test_annotations):
            seps_included = seps_included or " | " in p # check it predicts separators
            pred_correct = pred_correct or any(li in p for li in l.split(" | ")) # at least one extraction is predicted correctly.

        self.assertTrue(seps_included)
        self.assertTrue(pred_correct)

    def test_t5(self):
        self.check_embeddings_equal(HFT5, "t5-base")


    def test_albert(self):
        self.check_embeddings_equal(HFAlbert, "albert-base-v2")

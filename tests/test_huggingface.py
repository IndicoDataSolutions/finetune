import unittest
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

from finetune import SequenceLabeler
from finetune.util.tokenization import WEIRD_TEXT
from finetune.base_models.huggingface.models import (
    HFBert,
    HFElectraGen,
    HFElectraDiscrim,
    HFXLMRoberta,
    HFT5,
    HFAlbert,
)


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
        finetune_seq_features = finetune_model.featurize_sequence([self.text])
        hf_seq_features = self.huggingface_embedding(self.text, hf_model_path)
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
            outputs = model(input_ids, decoder_input_ids=input_ids)
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

    def test_t5(self):
        self.check_embeddings_equal(HFT5, "t5-base")

    def test_albert(self):
        self.check_embeddings_equal(HFAlbert, "albert-base-v2")

import unittest
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

from finetune import SequenceLabeler
from finetune.base_models.huggingface.models import HFBert, HFElectraGen, HFElectraDiscrim, HFXLMRoberta


class TestHuggingFace(unittest.TestCase):
    def setUp(self):
        self.text = """
        The quick brown fox jumps over the lazy dog. Il ne faut pas se fier aux apparences.
        Año nuevo, vida nueva. Das ist mir Wurst.
        """

    def check_embeddings_equal(self, finetune_base_model, hf_model_path):
        finetune_model = SequenceLabeler(
            base_model=finetune_base_model, train_embeddings=False, n_epochs=1, batch_size=1)
        finetune_model.fit([self.text], [[{'start': 0, 'end': 4, 'label': 'class_a'}]])
        np.testing.assert_array_almost_equal(
            finetune_model.featurize_sequence([self.text]),
            self.huggingface_embedding(self.text, hf_model_path),
            decimal=5
        )

    def huggingface_embedding(self, text, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModel.from_pretrained(model_path)
        input_ids = tf.constant(tokenizer.encode(self.text))[None, :]  # Batch size 1
        outputs = model(input_ids)  # outputs is tuple where first element is sequence features
        last_hidden_states = outputs[0]
        return last_hidden_states.numpy()

    def test_xlm_roberta(self):
        self.check_embeddings_equal(HFXLMRoberta, "jplu/tf-xlm-roberta-base")

    def test_bert(self):
        self.check_embeddings_equal(HFBert, "bert-base-uncased")

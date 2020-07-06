import unittest
import numpy as np
from transformers import AutoTokenizer, TFAutoModel

from finetune.base_models.huggingface.models import HFBert, HFElectraGen, HFElectraDiscrim, HFXLMRoberta


def huggingface_embedding(text, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFAutoModel.from_pretrained(model_path)
    input_ids = tf.constant(tokenizer.encode(self.text))[None, :]  # Batch size 1
    outputs = model(input_ids)  # outputs is tuple where first element is sequence features
    last_hidden_states = outputs[0]
    return tf.make_ndarray(last_hidden_states)

class TestHuggingFace(unittest.TestCase):
    def setUp(self):
        self.text = "The quick brown fox jumps over the lazy dog"

    def test_xlm_roberta(self):
        finetune_model = Classifier(base_model=HFXLMRoberta)
        np.testing.assert_array_almost_equal(
            finetune_model.featurize_sequence(self.text)
            huggingface_embedding(self.text, "jplu/tf-xlm-roberta-base")
        )

    def test_electra_discriminator(self):
        finetune_model = Classifier(base_model=HFElectraDiscrim)
        np.testing.assert_array_almost_equal(
            finetune_model.featurize_sequence(self.text)
            huggingface_embedding(self.text, "google/electra-base-discriminator")
        )

    def test_electra_generator(self):
        finetune_model = Classifier(base_model=HFElectraGen)
        np.testing.assert_array_almost_equal(
            finetune_model.featurize_sequence(self.text)
            huggingface_embedding(self.text, "google/electra-base-generator")
        )

import unittest
import os.path

import numpy as np
from tensorflow.data import Dataset

from finetune.encoding.input_encoder import EncodedOutput
from finetune.model import PredictMode
from finetune.base_models import GPT, GPT2, BERT
from finetune import Classifier, MultiFieldClassifier

DIRECTORY = os.path.abspath(os.path.dirname(__file__))


class TestActivationParity(unittest.TestCase):

    MULTIFIELD_TEST_DATA = [[
        "Rick grew up in a troubled household. He never found good support in family, "
        "and turned to gangs. It wasn't long before Rick got shot in a robbery. The "
        "incident caused him to turn a new leaf.", 
        "He is happy now."
    ]]
    TEST_DATA = ["this is a test"]

    def test_gpt_multifield_parity(self):
        model = MultiFieldClassifier()
        np.testing.assert_allclose(
            model.featurize(self.MULTIFIELD_TEST_DATA)[0], 
            np.load(
                os.path.join(
                    DIRECTORY, 
                    'data/test-gpt-multifield-activations.npy'
                )
            ),
            atol=1e-1
        )

    def test_gpt2_featurize(self):
        model = Classifier(base_model=GPT2)
        
        def dataset_encoded():
            yield {"tokens": arr_encoded.token_ids, "mask": arr_encoded.mask}

        def get_input_fn():
            types, shapes = model.input_pipeline.feed_shape_type_def()
            tf_dataset = Dataset.from_generator(dataset_encoded, types[0], shapes[0])
            return tf_dataset.batch(1)

        encoded = model.input_pipeline.text_encoder._encode(self.TEST_DATA)
        encoded = EncodedOutput(token_ids=encoded.token_ids[0])
        estimator, hooks = model.get_estimator(force_build_lm=False)
        predict = estimator.predict(
            input_fn=get_input_fn, predict_keys=[PredictMode.SEQUENCE], hooks=hooks
        )
        arr_encoded = model.input_pipeline._array_format(encoded)
        sequence_features = next(predict)[PredictMode.SEQUENCE]

        np.testing.assert_allclose(
            sequence_features[:len(encoded.token_ids),:],
            np.load(
                os.path.join(
                    DIRECTORY, 
                    'data/test-gpt2-activations.npy'
                )
            ),
            atol=1e-1
        )

    def test_bert_featurize(self):
        model = Classifier(base_model=BERT)
        np.testing.assert_allclose(
            model.featurize(self.TEST_DATA)[0], 
            np.load(
                os.path.join(
                    DIRECTORY, 
                    'data/test-bert-activations.npy'
                )
            ),
            atol=1e-1
        )

    
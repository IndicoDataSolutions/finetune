import os
import glob
import unittest
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from finetune import MaskedLanguageModel, Classifier
from finetune.errors import FinetuneError
from finetune.base_models import GPT2, BERT, RoBERTa


class TestMaskedLanguageModel(unittest.TestCase):
    n_sample = 20
    dataset_path = os.path.join("Data", "Classify", "SST-binary.csv")

    @classmethod
    def _download_sst(cls):
        """
        Download Stanford Sentiment Treebank to data directory
        """
        path = Path(cls.dataset_path)
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME,
        )

    @classmethod
    def setUpClass(cls):
        cls._download_sst()

    def setUp(self):
        self.dataset = pd.read_csv(self.dataset_path, nrows=self.n_sample * 3)
        try:
            os.mkdir("tests/saved-models")
        except FileExistsError:
            warnings.warn(
                "tests/saved-models still exists, it is possible that some test is not cleaning up properly."
            )

    def test_fit_predict_bert(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        model = MaskedLanguageModel(base_model=BERT)
        save_file = "tests/saved-models/test-mlm"
        sample = self.dataset.sample(n=self.n_sample)
        model.fit(sample.Text)

        with self.assertRaises(FinetuneError):
            predictions = model.predict(sample.Text)

    def test_fit_predict_bert_generators(self):
        """                                                                                                                                            
        Ensure saving + loading does not cause errors                                                                                                  
        Ensure saving + loading does not change predictions                                                                                            
        """
        model = MaskedLanguageModel(base_model=BERT, dataset_size=self.n_sample)
        save_file = "tests/saved-models/test-mlm"
        sample = self.dataset.sample(n=self.n_sample)
        def generator_fn():
            for x in sample.Text:
                yield {"X": x}
        
        model.fit(generator_fn)

        with self.assertRaises(FinetuneError):
            predictions = model.predict(generator_fn)


    def test_create_new_base_model_roberta(self):
        """
        Ensure we can fit / save / re-load RoBERTa models trained with MLM objective
        """
        model = MaskedLanguageModel(base_model=RoBERTa)
        save_file = "bert/test-mlm.jl"
        sample = self.dataset.sample(n=self.n_sample)
        model.fit(sample.Text)

        with self.assertRaises(Exception):
            predictions = model.predict(sample.Text)

        model.create_base_model(save_file)
        model = Classifier(base_model=RoBERTa, base_model_path=save_file)
        model.fit(sample.Text.values, sample.Target.values)

        predictions = model.predict(sample.Text.values)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

    def test_exception_gpt2(self):
        """
        Ensure an explicit exception is raised for GPT2 w/ MLM 
        """
        with self.assertRaises(FinetuneError):
            model = MaskedLanguageModel(base_model=GPT2)

    def tearDown(self):
        for f in glob.glob('bert/test/test-mlm.*'):
            os.remove(f)

if __name__ == '__main__':
    unittest.main()

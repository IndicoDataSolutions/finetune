import os
import unittest
import warnings

# prevent excessive warning logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split

from finetune import MultiTask, Classifier, Comparison
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank
from finetune.datasets.quora_similarity import QuoraDuplicate
from finetune.config import finetune_model_path
SST_FILENAME = "SST-binary.csv"


class TestMTL(unittest.TestCase):

    def test_fit_predict(self):
        dataset = StanfordSentimentTreebank(nrows=50).dataframe
        q_dataset = QuoraDuplicate(nrows=50).dataframe

        model = MultiTask(
            tasks={
                "sst": Classifier,
                "qqp": Comparison
            },
            n_epochs=2,
            optimizer="AdamaxW",
            max_length=200,
        )

        q_X1, q_X2, q_Y = q_dataset.Text1.values, q_dataset.Text2.values, q_dataset.Target.values

        trainX, testX, trainY, testY = train_test_split(
            dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42
        )

        model.fit(
            {
                "sst": trainX,
                "qqp": list(zip(q_X1, q_X2)),
            },
            {
                "sst": trainY,
                "qqp": q_Y,
            }
        )

        model.featurize(
            {
                "sst": testX,
                "qqp": list(zip(q_X1, q_X2))[:10],
            }
        )

        preds = model.predict(
            {
                "sst": testX,
                "qqp": list(zip(q_X1, q_X2))[:10],
            }
        )
        self.assertIn("sst", preds)
        self.assertIn("qqp", preds)

        model.create_base_model("./test_base_mtl.jl", exists_ok=True)
        model = Classifier(base_model_path="./test_base_mtl.jl", max_length=200)
        model.fit(trainX, trainY)

        os.remove(finetune_model_path("./test_base_mtl.jl"))

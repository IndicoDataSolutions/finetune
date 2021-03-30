
import numpy as np

from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from finetune import Classifier
from finetune.datasets import Dataset, generic_download
from finetune.base_models import LongDocBERT, BERT
from finetune.base_models.gpt.model import GPTModel
from finetune.base_models.oscar.model import GPCModel
#logging.basicConfig(level=logging.DEBUG)

from sklearn.metrics import classification_report

SST_FILENAME = "SST-binary.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
CHECKSUM = "02136b7176f44ff8bec6db2665fc769a"


class StanfordSentimentTreebank(Dataset):

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    def md5(self):
        return CHECKSUM

    def download(self):
        """
        Download Stanford Sentiment Treebank to data directory
        """
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME
        )


if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    # model = Classifier(
    #     base_model=BERT,
    #     chunk_long_sequences=False,
    #     max_length=512,
    #     visible_gpus=["0"],
    #     debugging_logs=True,
    #     num_layers_trained=0,
    #     train_embeddings=False,
    #     lr=0.0005,
    # )
    long_model = Classifier(
        base_model=LongDocBERT,
        visible_gpus=["0"],
        max_length=512,
        chunk_size=128,
        debugging_logs=True,
        lr=0.001,
        chunk_pool_fn="mean",
        n_epochs=512,
        batch_size=8,
    )

    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)
    # model.fit(trainX, trainY)
    long_model.fit(trainX[:8], trainY[:8])
    # preds = model.predict(testX)
    long_preds = long_model.predict(trainX[:8])
    # print(preds, testY)
    # print(classification_report(testY, preds))
    print(classification_report(trainY[:8], long_preds))

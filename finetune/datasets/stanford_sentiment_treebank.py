
import numpy as np

from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from finetune import Classifier
from finetune.datasets import Dataset, generic_download
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
    model = Classifier(
        debugging_logs=True,
        summarize_grads=True,
#        val_interval=1000,
    )
    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)
    model.fit(trainX, trainY)
    preds = model.predict(testX)
    print(preds, testY)
    print(classification_report(testY, preds))


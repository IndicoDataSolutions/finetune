import os
import logging
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import Classifier
from finetune.datasets import Dataset, generic_download

logging.basicConfig(level=logging.DEBUG)

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
            filename=SST_FILENAME,
        )


if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    model = Classifier(verbose=True, n_epochs=2, val_size=0.01, val_interval=10, visible_gpus=[], tensorboard_folder='.tensorboard', max_length=32, chunk_long_sequence=True)
    trainX, testX, trainY, testY = train_test_split(dataset.Text, dataset.Target, test_size=0.3, random_state=42)
    model.fit(trainX.values, trainY.values)
    accuracy = np.mean(model.predict(testX.values) == testY.values)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

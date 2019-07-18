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
            filename=SST_FILENAME
        )


if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=200).dataframe
    pre_train_generator = lambda: iter(StanfordSentimentTreebank(nrows=5000).dataframe.Text.values)
    model = Classifier(n_epochs=3, batch_size=2, lr_warmup=0.1, tensorboard_folder='.tensorboard')
    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)
    model.config.dataset_size = 5000
    model.config.val_size = 100
    model.config.val_interval = 1000
    model.config.batch_size = 5
    model.fit(pre_train_generator)
    model.config.val_size = None
    model.config.val_interval = None
    #model.config.dataset_size = 1000 # This is automatically set as trainX has len
    model.config.batch_size = 2
    model.fit(trainX, trainY)
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

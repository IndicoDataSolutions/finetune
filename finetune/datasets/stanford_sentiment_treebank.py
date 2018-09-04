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
    np.random.seed(42)
    dataset = StanfordSentimentTreebank(nrows=500).dataframe
    for beta_coef in np.linspace(0.0, 0.2, num=5):
        model = Classifier(
            verbose=True, 
            n_epochs=5, 
            val_size=0., 
            val_interval=2**32, 
            lr_warmup=0.1, 
            tensorboard_folder='.tensorboard',
            max_length=128,
            beta_coef=beta_coef
        )
        print("Configuration: {}".format(model.config))
        trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.5, random_state=42)
        n_train = trainY.shape[0]
        percent_corruption = 0.25
        print("Percent corruption: {:0.2f}".format(percent_corruption))
        n_corrupted = int(n_train * percent_corruption)
        corrupt_indexes = np.random.choice(list(range(n_train)), size=n_corrupted)
        trainY[corrupt_indexes] = np.invert(trainY[corrupt_indexes])
        model.fit(trainX, trainY)
        accuracy = np.mean(model.predict(testX) == testY)
        print('Test Accuracy: {:0.2f}'.format(accuracy))

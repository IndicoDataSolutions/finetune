import os
import tempfile
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import enso
from enso.download import generic_download
from sklearn.model_selection import train_test_split

from finetune import Classifier
from finetune.datasets import Dataset

logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "SST-binary.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)

class StanfordSentimentTreebank(Dataset):

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    def download(self):
        """
        Download Stanford Sentiment Treebank to enso `data` directory
        """
        path = Path(self.filename)
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME
        )


if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=1500).dataframe
    model = LanguageModelClassifier(val_size=100, val_interval=5000, verbose=True)
    trainX, testX, trainY, testY = train_test_split(dataset.Text, dataset.Target, test_size=0.3, random_state=42)
    model.fit(trainX, trainY)
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

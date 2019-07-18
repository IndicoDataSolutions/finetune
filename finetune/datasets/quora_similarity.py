import os
import tempfile
import hashlib
import logging
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import Comparison
from finetune.datasets import Dataset, comparison_download

logging.basicConfig(level=logging.DEBUG)

QUORA_SIMILARITY = "Quora.csv"
DATA_PATH = os.path.join('Data', 'Similarity', QUORA_SIMILARITY)
CHECKSUM = "6f2642eec22c1f0f3fc2fb2dfb9dfc38899d7c58bfc9f4c7e4d1a7e3cea15a29"


class QuoraDuplicate(Dataset):
    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    @property
    def md5(self):
        return CHECKSUM

    def download(self):
        """
        Download quora duplicate questions dataset.
        """
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison_download(
            url="https://s3.amazonaws.com/enso-data/Quora.csv",
            text_column1="Text1",
            text_column2="Text2",
            target_column="Target",
            filename=QUORA_SIMILARITY
        )

if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = QuoraDuplicate(nrows=5000).dataframe
    model = Comparison(n_epochs=1)
    trainX1, testX1, trainX2, testX2, trainY, testY = train_test_split(
        dataset.Text1.values, dataset.Text2.values, dataset.Target.values,
        test_size=0.3, random_state=42
    )
    model.fit(list(zip(trainX1, trainX2)), trainY)
    accuracy = np.mean(model.predict(list(zip(testX1, testX2))) == testY)
    class_balance = np.mean(testY)
    print('Test Accuracy: {:0.2f} for a {:0.2f} class balance'.format(accuracy, class_balance))

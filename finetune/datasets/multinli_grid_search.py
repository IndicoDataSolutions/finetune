import os
import logging
from pathlib import Path
import hashlib

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from finetune import MultifieldClassifier
from finetune.datasets import Dataset
from finetune.config import get_default_config

logging.basicConfig(level=logging.DEBUG)

FILENAME = "multinli.dev.csv"
DATA_PATH = os.path.join('Data', 'Entailment', FILENAME)
CHECKSUM = "4837f671a2ee1042f3d308de5352b58e"


class MultiNLI(Dataset):

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)


    @property
    def md5(self):
        return CHECKSUM

    def download(self):
        """
        Download Stanford Sentiment Treebank to data directory
        """
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        remote_url = "https://s3.amazonaws.com/enso-data/multinli.dev.csv"

        response = requests.get(remote_url)
        open(DATA_PATH, 'wb').write(response.content)



if __name__ == "__main__":
    # Train and evaluate on MultiNLI
    dataset = MultiNLI(nrows=10).dataframe
    trainX1, testX1, trainX2, testX2, trainY, testY = train_test_split(
        dataset.x1, dataset.x2, dataset.target, test_size=0.3, random_state=42
    )
    base_conf = get_default_config()
    res = MultifieldClassifier.finetune_grid_search_cv([dataset.x1, dataset.x2], dataset.target, n_splits=2, config=base_conf,
                                                    eval_fn=lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)),
                                                    test_size=0.1)

    model = MultifieldClassifier(res)
    model.fit(trainX1, trainX2, Y=trainY)
    acc = np.mean(np.asarray(model.predict(testX1, testX2)) == np.asarray(testY))
    print('Test Accuracy: {:0.2f} with config {}'.format(acc, res))
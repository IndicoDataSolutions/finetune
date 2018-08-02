import os
import logging
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import MultiLabelClassifier
from finetune.datasets import Dataset, generic_download

logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "mcdonalds_yelp.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
CHECKSUM = "02136b7176f44ff8bec6db2665fc769a"


class MCDonaldsSentiment(Dataset):

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    def download(self):
        """
        Download Stanford Sentiment Treebank to data directory
        """
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://www.figure-eight.com/wp-content/uploads/2016/03/McDonalds-Yelp-Sentiment-DFE.csv",
            text_column="review",
            target_column="policies_violated",
            filename=SST_FILENAME,
            target_transformation=lambda x: x.split("\n")
        )


if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = MCDonaldsSentiment(nrows=1000).dataframe
    print(dataset.Target)
    model = MultiLabelClassifier(verbose=True, n_epochs=2)
    trainX, testX, trainY, testY = train_test_split(dataset.Text, dataset.Target, test_size=0.3, random_state=42)
    model.fit(trainX, trainY)
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

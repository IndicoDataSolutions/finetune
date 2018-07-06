import os
import unittest
import logging
from copy import copy
from pathlib import Path

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
import enso
import tempfile
from enso.download import generic_download
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from finetune import LanguageModelClassifier
import logging

logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "SST-binary.csv"

n_sample = 100
n_hidden = 768
dataset_path = os.path.join(enso.config.DATA_DIRECTORY, 'Classify', 'SST-binary.csv')


def _download_sst():
    """
    Download Stanford Sentiment Treebank to enso `data` directory
    """
    path = Path(dataset_path)
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    generic_download(
        url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
        text_column="Text",
        target_column="Target",
        filename=SST_FILENAME
    )


_download_sst()
dataset = pd.read_csv(dataset_path, nrows=1500) # NOT THE WHOLE DATASET
save_file_autosave = tempfile.mkdtemp()
model = LanguageModelClassifier(verbose=True, autosave_path=save_file_autosave)
Xtr, Xte, ttr, tte = train_test_split(dataset.Text, dataset.Target, test_size=0.3, random_state=12345)
model.fit(Xtr, ttr, val_size=100, val_interval=5000)
model.save(save_file_autosave)  # Overwrite the early stopping.
accuracy = np.mean(model.predict(Xte) == tte)
print('Test Accuracy is', accuracy)

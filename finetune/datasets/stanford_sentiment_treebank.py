import os
import logging
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import Classifier
from finetune.datasets import Dataset, generic_download
from finetune.base_models.gpt.model import GPTModel
from finetune.base_models.gpt2.model import GPT2Model
from finetune.base_models.gpc.model import GPCModel
import joblib as jl
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
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)
    feat_modes = ["final_state", "clf_tok", "mean_state", "mean_tok", "max_state", "max_tok"] 
    for l2 in [0.0, 0.001, 0.01, 0.1]:
        for prefit_init in [False]:#True, False]:
            for lr_warmup in [0.0, 0.1, 0.3]:
                for batch_size in [2, 3, 4]:
                    for lr in [1e-4]:
                        for epoch in [3]:
                            for feat_mode in ["clf_tok"]:
                                model = Classifier(
                                    max_length=64,
                                    n_epochs=epoch, 
                                    batch_size=batch_size, 
                                    lr_warmup=lr_warmup,
                                    val_size=0,
                                    lr=lr,
                                    base_model=GPCModel,
                                    base_model_path="conv_base_30jun.jl",
                                    xla=False,
                                    keep_best_model=False,#True,
                                    l2_reg=l2,
                                    prefit_init=prefit_init,
                                    feat_mode=feat_mode
                                )
                                print(dict(
                                    n_epochs=epoch,
                                    batch_size=batch_size,
                                    lr_warmup=lr_warmup,
                                    val_size=0,
                                    lr=lr,
                                    l2_reg=l2,
                                    prefit_init=prefit_init,
                                    feat_mode=feat_mode
                                )
                                )

                                model.fit(trainX, trainY)
                                accuracy = np.mean(model.predict(testX) == testY)

#                                model = Classifier(
#                                    max_length=64,
#                                    n_epochs=epoch,
#                                    batch_size=batch_size,
#                                    lr_warmup=lr_warmup,
#                                    val_size=0,
#                                    lr=lr,
#                                    base_model=GPCModel,
#                                    base_model_path="conv_base_30jun.jl",#"conv25days.jl",
#                                    xla=False,
#                                    keep_best_model=False,#True,
#                                    l2_reg=l2,
#                                    prefit_init=prefit_init,
#                                    feat_mode=feat_mode
#                                )
#                                model.fit(trainX[:100], trainY[:100])
#                                accuracy_100 = np.mean(model.predict(testX) == testY)
                                print('Test Accuracy 1000: {:0.2f}, 100: {:0.2f}'.format(accuracy, -1))

import os
import logging

import numpy as np
import pandas as pd


from finetune import Classifier

logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "SST-binary.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
CHECKSUM = "02136b7176f44ff8bec6db2665fc769a"




if __name__ == "__main__":
    # Train and evaluate on SST
    train = pd.read_csv("/root/code/indico/finetune/Data/glue_data/SST-2/train.tsv", sep="\t")
    dev = pd.read_csv("/root/code/indico/finetune/Data/glue_data/SST-2/dev.tsv", sep="\t")
    model = Classifier(verbose=True, n_epochs=3, batch_size=4, lr_warmup=0.1, tensorboard_folder='sst_full_run', summarize_grads=True)
    model.fit(train.sentence.values, train.label.values)
    accuracy = np.mean(model.predict(dev.sentence.values) == dev.label.values)
    print('Test Accuracy: {:0.2f}'.format(accuracy))
    model.save("SST_FULL_MODEL.jl")

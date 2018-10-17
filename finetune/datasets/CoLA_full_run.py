import os
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from finetune import Classifier

logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "SST-binary.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
CHECKSUM = "02136b7176f44ff8bec6db2665fc769a"




if __name__ == "__main__":
    # Train and evaluate on SST
    train = pd.read_csv("/root/code/indico/finetune/Data/glue_data/CoLA/train.tsv", sep="\t", header=None, names=["not_sure", "label","_", "sentence"])
    dev = pd.read_csv("/root/code/indico/finetune/Data/glue_data/CoLA/dev.tsv", sep="\t", header=None, names=["not_sure", "label","_", "sentence"])
    model = Classifier(verbose=True, n_epochs=3, batch_size=2, lr_warmup=0.1, tensorboard_folder='sst_full_run/CoLA', summarize_grads=True, oversample=True, eval_acc=True)
    print(dev.label.values)
    model.fit(train.sentence.values, train.label.values)
    #model = Classifier.load("CoLA_FULL_MODEL.jl")
    preds = model.predict(dev.sentence.values)
    print(np.mean(preds))
    accuracy = matthews_corrcoef(dev.label.values, preds)
    print('Test Matthews Corr: {:0.2f}'.format(accuracy))
    model.save("CoLA_FULL_MODEL.jl")

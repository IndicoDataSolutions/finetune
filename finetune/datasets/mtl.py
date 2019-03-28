import os
import logging
from pathlib import Path
import json

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import MultiTask, Classifier, Comparison
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank
from finetune.datasets.quora_similarity import QuoraDuplicate
from finetune.datasets.reuters import Reuters
logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "SST-binary.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
CHECKSUM = "02136b7176f44ff8bec6db2665fc769a"

if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    q_dataset = QuoraDuplicate(nrows=1000).dataframe
    r_dataset = Reuters().dataframe
    r_dataset['annotations'] = [json.loads(annotation) for annotation in r_dataset['annotations']]
    r_trainX, r_testX, r_trainY, r_testY = train_test_split(
        r_dataset.texts.values,
        r_dataset.annotations.values,
        test_size=0.3,
        random_state=42
    )
    
    model = MultiTask(tasks={
        "sst": Classifier,
        "qqp": Comparison
    }, debugging_logs=True, lr=6.25e-6, n_epochs=2)

    q_trainX1, q_testX1, q_trainX2, q_testX2, q_trainY, q_testY = train_test_split(
        q_dataset.Text1.values, q_dataset.Text2.values, q_dataset.Target.values,
        test_size=0.3, random_state=42
    )
    trainX, testX, trainY, testY = train_test_split(
        dataset.Text.values, dataset.Target.values, test_size=0.3,  random_state=42
    )
    model.fit(
        {
            "sst": trainX,
            "qqp": list(zip(q_trainX1, q_trainX2)),
        },
        {
            "sst": trainY,
            "qqp": q_trainY,
        }
    )
    model.create_base_model("mtl_sst_qqp_test.jl", exists_ok=True)
    model = Classifier(base_model_path="mtl_sst_qqp_test.jl")
    model.fit(trainX, trainY)
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

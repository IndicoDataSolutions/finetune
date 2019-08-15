import logging

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import Classifier
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank
from finetune.base_models.bert.model import BERTModelCased
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    model = Classifier(val_size=0.0, max_length=512, base_model=BERTModelCased, batch_size=32, low_memory_mode=True, accum_steps=3)
    print(model.config.base_model_path)
    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)
    model.fit(trainX, trainY)
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

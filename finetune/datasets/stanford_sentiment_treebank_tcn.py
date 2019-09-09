import logging

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import Classifier
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank
from finetune.base_models import TCN
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    model = Classifier(
        # val_size=0.,
        max_length=64,
        base_model=TCN,
        # early_stopping_steps=100,
        n_epochs=5, 
        keep_best_model=True,
        debugging_logs=True
    )
    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)
    model.fit(trainX, trainY)
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

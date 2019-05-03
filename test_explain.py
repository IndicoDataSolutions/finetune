import ipdb
import tensorflow as tf
from sklearn.model_selection import train_test_split

from finetune import Classifier
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank


if __name__ == "__main__":
    dataset = StanfordSentimentTreebank(nrows=200).dataframe
    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)
    
    model = Classifier(batch_size=2, val_size=0.)
    model.fit(trainX, trainY)
    print(model.explain(testX, testY))


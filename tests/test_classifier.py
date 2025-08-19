import pytest
import time

import numpy as np
from sklearn.metrics import recall_score

from finetune import Classifier
from finetune.errors import FinetuneError

SST_FILENAME = "SST-binary.csv"

def test_multiple_models_fit_predict(get_untrained_classifier, classification_text_sample):
    model = get_untrained_classifier()
    train_sample = classification_text_sample()
    valid_sample = classification_text_sample()
    model.fit(train_sample.Text.values, train_sample.Target.values)
    model.predict(valid_sample.Text.values)
    model.close()

    model2 = get_untrained_classifier()
    model2.fit(train_sample.Text.values, train_sample.Target.values)
    model2.predict(valid_sample.Text.values)

def test_cached_predict(trained_classifier, classification_text_sample):
    """
    Ensure second call to predict is faster than first
    """
    valid_sample = classification_text_sample()

    start = time.time()
    uncached_predictions = trained_classifier.predict(valid_sample.Text[:1].values)
    first = time.time()
    cached_predictions = trained_classifier.predict(valid_sample.Text[:1].values)
    second = time.time()

    first_prediction_time = first - start
    second_prediction_time = second - first
    assert second_prediction_time < first_prediction_time / 2.0
    assert uncached_predictions == cached_predictions

def test_train_error_on_different_lengths(get_untrained_classifier, classification_text_sample):
    model = get_untrained_classifier()
    train_sample = classification_text_sample()

    with pytest.raises(FinetuneError):
        model.fit(train_sample.Text, train_sample.Target[:1])

    model.fit(train_sample.Text.values, train_sample.Target.values)

def test_class_weights(get_untrained_classifier, trained_classifier, classification_text_sample):
    # testing class weights
    train_sample = classification_text_sample()
    valid_sample = classification_text_sample()
    predictions = trained_classifier.predict(valid_sample.Text.values)
    recall = recall_score(valid_sample.Target.values, predictions, pos_label=1)
    model = get_untrained_classifier(class_weights={1: 100})
    model.fit(train_sample.Text.values, train_sample.Target.values)
    predictions = model.predict(valid_sample.Text.values)
    new_recall = recall_score(valid_sample.Target.values, predictions, pos_label=1)
    assert new_recall >= recall

def test_chunk_long_sequences(get_untrained_classifier):
    test_sequence = [
        "This is a sentence to test chunk_long_sequences in classification. " * 20,
        "Another example so now there are two different classes in the test. " * 20,
    ]
    labels = ["a", "b"]
    model = get_untrained_classifier(chunk_long_sequences=True, max_length=10)
    model.finetune(test_sequence * 10, labels * 10)
    predictions = model.predict(test_sequence * 10)
    probas = model.predict_proba(test_sequence * 10)
    assert len(predictions) == 20
    assert len(probas[0]) == 2
    np.testing.assert_almost_equal(np.sum(list(probas[0].values())), 1, decimal=4)

def test_fit_predict_batch_size_1(get_untrained_classifier, sst_dataset):
    model = get_untrained_classifier(batch_size=1, predict_batch_size=1)
    train_sample = sst_dataset.sample(n=20)
    valid_sample = sst_dataset.sample(n=20)
    model.fit(train_sample.Text.values, train_sample.Target.values)
    model.predict(valid_sample.Text.values)

def test_save_load(trained_classifier, classification_text_sample):
    """
    Ensure saving + loading does not cause errors
    Ensure saving + loading does not change predictions
    """
    save_file = "tests/saved-models/test-save-load"
    valid_sample = classification_text_sample()
    predictions = trained_classifier.predict(valid_sample.Text)
    model = Classifier.load(save_file)
    new_predictions = model.predict(valid_sample.Text)
    assert predictions == new_predictions

def test_featurize(trained_classifier, classification_text_sample):
    """
    Ensure featurization returns an array of the right shape
    Ensure featurization is still possible after fit
    """
    features = trained_classifier.featurize(classification_text_sample().Text)
    assert features.shape == (len(classification_text_sample()), trained_classifier.config.n_embed)
    sequence_features = trained_classifier.featurize_sequence(classification_text_sample().Text)
    assert len(sequence_features.shape) == 3
    assert sequence_features.shape[0] == len(classification_text_sample())
    assert sequence_features.shape[2] == trained_classifier.config.n_embed

def test_reasonable_predictions(get_untrained_classifier):
    """
    Ensure model converges to a reasonable solution for a trivial problem
    """
    model = get_untrained_classifier(n_epochs=5)
    n_duplicates = 5
    trX = ["cat", "kitten", "feline", "meow", "kitty"] * n_duplicates + [
        "finance",
        "investment",
        "investing",
        "dividends",
        "financial",
    ] * n_duplicates
    trY = ["cat"] * (len(trX) // 2) + ["finance"] * (len(trX) // 2)
    model.fit(trX, trY)
    predY = model.predict(trX)
    assert predY == trY

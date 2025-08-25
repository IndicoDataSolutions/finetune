import sys
import pytest
import time
from collections import Counter
import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score

from finetune import Classifier
from finetune.errors import FinetuneError

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
    trained_classifier.close() # Close the model, because it is shared between tests it may have already been loaded.
    start = time.time()
    uncached_predictions = trained_classifier.predict(valid_sample.Text[:1].values)
    first = time.time()
    cached_predictions = trained_classifier.predict(valid_sample.Text[:1].values)
    second = time.time()

    first_prediction_time = first - start
    second_prediction_time = second - first
    assert second_prediction_time < first_prediction_time / 1.5
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

def test_chunk_long_sequences(mock_get_keras_model, get_untrained_classifier):
    test_sequence = [
        "This is a sentence to test chunk_long_sequences in classification. " * 21,
        "Another example so now there are two different classes in the test. " * 21,
    ]
    labels = ["a", "b"]
    model = get_untrained_classifier(chunk_long_sequences=True, max_length=10, batch_size=1, predict_batch_size=1)
    model.finetune(test_sequence, labels)
    assert len(mock_get_keras_model) == 1
    assert len(mock_get_keras_model[0].fit_calls) == 1
    data_lengths = [data[0]["tokens"].shape[1] for data in mock_get_keras_model[0].fit_calls[0]["data"]]
    assert all(length == 10 for length in data_lengths) # For fit we pad everything to max_length for XLA.

    seq_1_num_tokens = len(model.input_pipeline.text_encoder.encode_multi_input([test_sequence[0]], max_length=sys.maxsize).tokens)
    seq_2_num_tokens = len(model.input_pipeline.text_encoder.encode_multi_input([test_sequence[1]], max_length=sys.maxsize).tokens)
    # Greater than for overlap and padding.
    assert sum(data_lengths) >= (seq_1_num_tokens + seq_2_num_tokens) * model.config.n_epochs

    class_counts = [tf.argmax(data[1][0], axis=-1).numpy().tolist() for data in mock_get_keras_model[0].fit_calls[0]["data"]]
    assert class_counts[0] == class_counts[1]

    predictions = model.predict(test_sequence)
    assert len(predictions) == 2
    assert all(pred in ["a", "b"] for pred in predictions)

    # We use pred batch size of 1. So this should be one call per chunk.
    assert len(mock_get_keras_model[0].predict_calls) >= (seq_1_num_tokens + seq_2_num_tokens) // 10

    probas = model.predict_proba(test_sequence)
    assert len(probas) == 2
    assert all(isinstance(proba, dict) for proba in probas)
    assert all(pytest.approx(np.sum(list(proba.values()))) == 1.0 for proba in probas)
    assert all(set(proba.keys()) == {"a", "b"} for proba in probas)


def test_save_load(trained_classifier, classification_text_sample, save_model_dir):
    """
    Ensure saving + loading does not cause errors
    Ensure saving + loading does not change predictions
    """
    save_path = save_model_dir / "test_save_load_classifier.jl"
    valid_sample = classification_text_sample()
    predictions = trained_classifier.predict(valid_sample.Text.values)
    trained_classifier.save(save_path)
    model = Classifier.load(save_path)
    new_predictions = model.predict(valid_sample.Text)
    assert predictions == new_predictions

def test_featurize(trained_classifier, classification_text_sample):
    """
    Ensure featurization returns an array of the right shape
    Ensure featurization is still possible after fit
    """
    text_sample = classification_text_sample().Text
    features = trained_classifier.featurize(text_sample)
    assert len(features) == len(text_sample)
    assert all(feature.shape == (trained_classifier.config.n_embed,) for feature in features)
    sequence_features = trained_classifier.featurize_sequence(text_sample)
    assert len(sequence_features) == len(text_sample)
    assert all(len(feature.shape) == 2 for feature in sequence_features)
    assert all(feature.shape[1] == trained_classifier.config.n_embed for feature in sequence_features)

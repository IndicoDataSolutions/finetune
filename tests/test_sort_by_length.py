import math
import random
import time

import numpy as np
import pytest

from finetune import Classifier


@pytest.mark.parametrize("do_sort_by_length", [True, False])
def test_sort_by_length(
    mock_get_keras_model, get_untrained_classifier, do_sort_by_length
):
    classifier = get_untrained_classifier(sort_by_length=do_sort_by_length)
    fake_data = (["A"] * 15 + [("B " * 120).strip()]) * 64
    classifier.fit(fake_data, ["A"] * len(fake_data))
    classifier.predict(fake_data)
    assert len(mock_get_keras_model) == 1
    model_mock = mock_get_keras_model[0]
    assert len(model_mock.predict_calls) == math.ceil(
        len(fake_data) / classifier.config.predict_batch_size
    )
    pred_lengths = []
    for pred_call in model_mock.predict_calls:
        pred_lengths.extend(pred_call["data"]["length"].numpy().tolist())
    assert set(pred_lengths) == {
        3,
        122,
    }  # 3 is the length of "A" and 122 is the length of "B " * 120 when EOS and BOS are added.
    if do_sort_by_length:
        assert pred_lengths == sorted(pred_lengths)
    else:
        assert pred_lengths != sorted(pred_lengths)


def test_sort(trained_classifier):
    zipped_data = [{"X": [1]}, {"X": [1, 2, 3, 4]}] * 50
    sorted_data, invert_idxs = trained_classifier._sort_by_length(zipped_data)
    assert sorted_data == sorted(sorted_data, key=lambda x: len(x["X"]))
    for i, idx in enumerate(invert_idxs):
        assert sorted_data[idx] == zipped_data[i]


@pytest.mark.parametrize("do_sort_by_length", [True, False])
def test_reconstruct(get_untrained_classifier, monkeypatch, do_sort_by_length):
    data = ["A " * i for i in range(1, 100)]
    random.shuffle(data)
    sorted_data = sorted(data, key=lambda x: len(x))
    mock_preds = list(range(len(data)))
    classifier = get_untrained_classifier(sort_by_length=do_sort_by_length)
    with monkeypatch.context() as m:
        m.setattr(classifier, "_predict", lambda x, **kwargs: mock_preds)
        preds = classifier.predict(data)

        if do_sort_by_length:
            assert preds != mock_preds
            for i, pred in enumerate(preds):
                # pred is the IDX of the pred at prediction time and i is the idx of the pred in the original data
                assert data[i] == sorted_data[pred]
        else:
            assert preds == mock_preds

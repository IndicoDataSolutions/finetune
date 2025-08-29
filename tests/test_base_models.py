import numpy as np
import pytest
from sequence_metrics.metrics import micro_f1
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from finetune import Classifier, SequenceLabeler
from finetune.base_models.textcnn.model import TextCNNModel

# The base model fixture handles running on all the valid base models.


def test_base_model_classifier(base_model, sst_dataset, save_model_dir):
    """
    A realistic test of the classifier on each base model.
    """
    allx = sst_dataset.Text.values
    ally = sst_dataset.Target.values
    trainx, testx, trainy, testy = train_test_split(allx, ally, test_size=0.1)
    model = Classifier(
        base_model=base_model, class_weights="sqrt", low_memory_mode=True
    )

    model.fit(trainx, trainy)
    predictions = model.predict(testx)

    model.save(save_model_dir / f"{base_model.__name__}_classifier.jl")
    post_save_preds = model.predict(testx)
    model.close()
    post_close_preds = model.predict(testx)
    model.close()  # close it so there are no memory contention issues

    model = Classifier.load(save_model_dir / f"{base_model.__name__}_classifier.jl")
    loaded_predictions = model.predict(testx)
    assert predictions == post_save_preds == post_close_preds == loaded_predictions

    accuracy = accuracy_score(testy, predictions)
    assert accuracy > 0.0  # Just make sure the model is doing something.

    probas = model.predict_proba(testx)
    assert len(probas) == len(testx)
    assert len(probas[0]) == 2
    for proba in probas:
        assert isinstance(proba, dict)
        np.testing.assert_approx_equal(np.sum(list(proba.values())), 1.0, significant=3)


def test_base_model_sequence_labeler(
    base_model, reuters_indico_sequence, save_model_dir
):
    """
    A realistic test of the sequence labeler on each base model.
    """
    if base_model is TextCNNModel:
        return pytest.skip(
            "Skipping textcnn model - is_bidirectional is False which means we end up with a very large attention block in the sequence labeler."
        )
    trainx, testx, trainy, testy = train_test_split(
        *reuters_indico_sequence, test_size=0.1
    )
    model = SequenceLabeler(
        base_model=base_model, class_weights="sqrt", low_memory_mode=True
    )
    model.fit(trainx, trainy)
    predictions = model.predict(testx)

    model.save(save_model_dir / f"{base_model.__name__}_sequence_labeler.jl")
    post_save_preds = model.predict(testx)
    model.close()
    post_close_preds = model.predict(testx)
    model.close()  # close it so there are no memory contention issues

    model = SequenceLabeler.load(
        save_model_dir / f"{base_model.__name__}_sequence_labeler.jl"
    )
    loaded_predictions = model.predict(testx)
    assert predictions == post_save_preds == post_close_preds == loaded_predictions

    micro_f1_score = micro_f1(testy, predictions)
    assert micro_f1_score > 0.0  # Just make sure the model is doing something

    probas = model.predict_proba(testx)
    assert len(probas) == len(testx)

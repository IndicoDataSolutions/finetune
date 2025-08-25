import os
import time
import pytest

from finetune.scheduler import Scheduler

@pytest.fixture(scope="module")
def models(saved_models_dir, trained_classifier, trained_annotation):
    model1 = os.path.join(saved_models_dir, "1.jl")
    model2 = os.path.join(saved_models_dir, "2.jl")
    trained_classifier.save(model1)
    trained_annotation.save(model2)
    yield model1, model2


def test_scheduler(models):
    model1, model2 = models
    shed = Scheduler()
    tic_1 = time.time()
    preds_m1 = shed.predict(model1, ["A"])  # May need isolation
    toc_1 = time.time()
    assert len(preds_m1) == 1
    assert isinstance(preds_m1[0], str) # classification
    preds_m2 = shed.predict(model2, ["A"])
    assert len(preds_m2) == 1
    assert isinstance(preds_m2[0], list) # Annotation - can't realy expect any labels though.
    tic_3 = time.time()
    shed.predict(model1, ["something else"])
    toc_3 = time.time()
    assert toc_1 - tic_1 > toc_3 - tic_3
    assert len(shed.loaded_models) == 2
    shed.close_all()
    assert len(shed.loaded_models) == 0
    shed.predict_proba(model1, ["A"])
    shed.featurize(model1, ["A"])
    shed.featurize_sequence(model1, ["A"])


def test_scheduler_max_models(models):
    model1, model2 = models
    shed = Scheduler(max_models=1)
    time_pre = time.time()
    pred1a = shed.predict(model1, ["A"])
    time_mid = time.time()
    pred1b = shed.predict(model1, ["A"])
    time_end = time.time()
    assert time_end - time_mid < time_mid - time_pre - 1
    assert pred1a == pred1b
    shed.predict(model2, ["A"])  # Load another model.
    assert len(shed.loaded_models) == 1

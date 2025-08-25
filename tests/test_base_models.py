
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from finetune import Classifier, SequenceLabeler


from sequence_metrics.metrics import micro_f1


# The base model fixture handles running on all the valid base models.

def test_base_model_classifier(base_model, sst_dataset, save_model_dir):
    """
    A realistic test of the classifier on each base model.
    """
    allx = sst_dataset.Text.values
    ally = sst_dataset.Target.values
    trainx, testx, trainy, testy = train_test_split(allx, ally, test_size=0.1)
    model = Classifier(base_model=base_model, class_weights="sqrt", low_memory_mode=True)
    
    model.fit(trainx, trainy)
    predictions = model.predict(testx)

    model.save(save_model_dir / f"{base_model.__name__}_classifier.jl")
    post_save_preds = model.predict(testx)
    model.close()
    post_close_preds = model.predict(testx)
    model.close() # close it so there are no memory contention issues

    model = Classifier.load(save_model_dir / f"{base_model.__name__}_classifier.jl")
    loaded_predictions = model.predict(testx)
    assert predictions == post_save_preds == post_close_preds == loaded_predictions

    # check that the accuracy of the model is > 70%
    accuracy = accuracy_score(testy, predictions)
    assert accuracy > 0.7

    probas = model.predict_proba(testx)
    assert len(probas) == len(testx)
    assert len(probas[0]) == 2
    assert np.all(np.sum(list(probas[0].values())) == 1)


def test_base_model_sequence_labeler(base_model, reuters_indico_sequence, save_model_dir):
    """
    A realistic test of the sequence labeler on each base model.
    """
    trainx, testx, trainy, testy = train_test_split(*reuters_indico_sequence, test_size=0.1)
    model = SequenceLabeler(base_model=base_model, class_weights="sqrt", low_memory_mode=True)
    model.fit(trainx, trainy)
    predictions = model.predict(testx)

    model.save(save_model_dir / f"{base_model.__name__}_sequence_labeler.jl")
    post_save_preds = model.predict(testx)
    model.close()
    post_close_preds = model.predict(testx)
    model.close() # close it so there are no memory contention issues

    model = SequenceLabeler.load(save_model_dir / f"{base_model.__name__}_sequence_labeler.jl")
    loaded_predictions = model.predict(testx)
    assert predictions == post_save_preds == post_close_preds == loaded_predictions

    # check that the micro f1 of the model is > 70%
    micro_f1_score = micro_f1(testy, predictions)
    assert micro_f1_score > 0.7

    probas = model.predict_proba(testx)
    assert len(probas) == len(testx)

    
    
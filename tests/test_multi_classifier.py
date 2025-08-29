import numpy as np
import pytest



def test_multilabel_fit_predict(get_untrained_multilabel_classifier, sst_dataset):
    """
    Ensure model training does not error out and returns correct types.
    """
    model = get_untrained_multilabel_classifier(n_epochs=5, l2_reg=0.0, clf_p_drop=0.0)
    train_sample = sst_dataset.sample(n=20)
    valid_sample = sst_dataset.sample(n=20)
    model.fit(train_sample.Text.values, [[t, 6, 3] for t in train_sample.Target.values])

    probabilities = model.predict_proba(valid_sample.Text.values)
    for proba in probabilities:
        assert isinstance(proba, dict)

    predictions = model.predict(valid_sample.Text.values)
    for prediction in predictions:
        assert isinstance(prediction[0], (str, int, np.integer))
        assert 3 in prediction
        assert 6 in prediction

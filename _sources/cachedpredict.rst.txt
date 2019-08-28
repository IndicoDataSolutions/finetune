Cached Prediction
=================

After fitting the model, call :py:func:`BaseModel.predict()` to infer on test data.

.. code-block:: python

    model = Classifier()
    model.fit(train_data, train_labels)
    model.predict(test_data)

To prevent recreating the tensorflow graph with each call to :py:func:`BaseModel.predict()`,
use the :py:func:`model.cached_predict()` context manager.

.. code-block:: python

    model = Classifier()
    model.fit(train_data, train_labels)
    with model.cached_predict():
        model.predict(test_data) # triggers prediction graph construction
        model.predict(test_data) # graph is already cached, so subsequence calls are faster
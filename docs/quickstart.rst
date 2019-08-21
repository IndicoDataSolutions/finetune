Finetune Quickstart Guide
=========================

Finetuning the base language model is as easy as calling :meth:`Classifier.fit`:

.. code-block:: python3

    model = Classifier()               # Load base model
    model.fit(trainX, trainY)          # Finetune base model on custom data
    predictions = model.predict(testX) # [{'class_1': 0.23, 'class_2': 0.54, ..}, ..]
    model.save(path)                   # Serialize the model to disk

Reload saved models from disk by using :meth:`Classifier.load`:

.. code-block:: python3

    model = Classifier.load(path)
    predictions = model.predict(testX)
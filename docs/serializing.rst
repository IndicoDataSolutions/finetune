Saving and Loading Models
=========================

You can use the :py:func:`BaseModel.save()` and :py:func:`.load()` methods to serialize and deserialize trained models.

.. code-block:: python

    model = Classifier()
    model.fit(train_data, train_labels)
    model.save(filepath)


.. code-block:: python 

    model = Classifier.load(filepath)

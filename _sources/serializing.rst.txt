Saving and Loading Models
=========================

You can use the :py:func:`BaseModel.save()` and :py:func:`.load()` methods to serialize and deserialize trained models. Note that if you use adapters or only train a subset of layers using the
:py:attr:`num_layers_trained` configuration flag, your save files will be much smaller. This is because Finetune only saves out weights that are different than the default weights of the base model.
If you are trying to optimize for save file size, check out :doc:`adapter`.

.. code-block:: python

    model = Classifier()
    model.fit(train_data, train_labels)
    model.save(filepath)


.. code-block:: python 

    model = Classifier.load(filepath)

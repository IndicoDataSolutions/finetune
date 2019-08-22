Process Long Sequences
======================

Most transformer-based models truncate their inputs to a maximum length, normally 512 tokens. Finetune transformers have functionality to handle
sequences of arbitrary length with the :py:attr:`chunk_long_sequences` flag. This moves a sliding window of length :py:attr:`max_length` across the input,
and takes the mean of the feature representations from the windows. Many classes that support :py:attr:`chunk_long_sequences`, such as Classifier and SequenceLabeler,
have it enabled by default.

.. code-block:: python

    model = Classifier(chunk_long_sequences=True)
    model.fit(train_data, train_labels)
    model.predict(test_data)
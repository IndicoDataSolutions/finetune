Using the SequenceLabeler Class
===============================

One of the dozen tasks our base models support is sequence labeling, where you label certain spans of text within a document rather than classifying the entire example. Labels for training
the SequenceLabeler are in the following format, as a list of lists of dictionaries:

.. code-block:: python
    
    # We include text, label, and start and end positions in our Y values. You do not need to create dictionaries for spans that have no label.
    # The text in the 'text' field must be equivalent to example[label['start']:label['end']]
    trainX = ['Intelligent process automation']
    trainY = [[
        {'text': 'Intelligent', 'capitalized': 'True', 'end': 11, 'start': 0, 'part_of_speech': 'ADJ'},
        {'text': 'process automation', 'start': 12, 'end': 30, 'part_of_speech': 'NOUN'}, 
    ]]

    from finetune import SequenceLabeler
    model = SequenceLabeler()
    model.fit(trainX, trainY)

    # Prediction outputs are in the same format as labels
    preds = model.predict(trainX)
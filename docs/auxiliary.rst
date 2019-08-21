Using Auxiliary Info
====================

Our base models can also process arbitrary auxiliary information in addition to text, such as style (bolding, italics, etc.), semantics (part-of-speech tags, sentiment tags), or other forms,
as long as they describe specific spans of text.

.. code-block:: python

    # First we define the extra features we will be providing, through a dictionary.
    # We do this by defining values and the defaults that tokens will receive if they are not explicitly labeled.
    # Auxiliary info can take the form of strings, booleans, floats, or ints.
    default = {'capitalized':False, 'part_of_speech':'unknown'}
    
    # Next we create context tags in a similar format to SequenceLabeling labels, as a list of lists of dictionaries:
    train_text = ['Intelligent process automation']
    train_context = [[
        {'text': 'Intelligent', 'capitalized': True, 'end': 11, 'start': 0, 'part_of_speech': 'ADJ'},
        {'text': 'process automation', 'capitalized': False, 'end': 30, 'start': 12, 'part_of_speech': 'NOUN'}, 
    ]]

    # Our input to the model is now a list containing the text, and the context
    trainX = [train_text, train_context]

    # Examples with no context must have an empty list as their context
    assert len(train_text) == len(train_context)

    # We indicate to the model that we are including auxiliary info by passing our default dictionary in with the default_context kwarg.
    model = Classifier(default_context=default)
    model.fit(trainX, trainY)
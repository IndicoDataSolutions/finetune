Using Auxiliary Info
====================

Our base models can also process arbitrary auxiliary information in addition to text, such as style (bolding, italics, etc.), semantics (part-of-speech tags, sentiment tags), or other forms,
as long as they describe specific spans of text.

.. code-block:: python

    # First we define the extra features we will be providing, through a dictionary.
    # Auxiliary info can take the form of booleans, floats, or ints. We currently cannot accept categorical inputs.

    # First we create context tags as a list of lists of dictionaries. Every token should have a context.
    train_text = ['Intelligent process automation']
    train_context = [[
        {'text': 'Intelligent', 'capitalized': True, 'end': 11, 'start': 0, 'part_of_speech': 'ADJ'},
        {'text': 'process automation', 'capitalized': False, 'end': 30, 'start': 12, 'part_of_speech': 'NOUN'}, 
    ]]

    # We then define the defaults that start and end tokens will receive.
    default = {'capitalized':False, 'part_of_speech':'unknown'}

    # We indicate to the model that we are including auxiliary info by passing our default dictionary in with the default_context kwarg.
    model = Classifier(default_context=default)
    # We finally pass in the context when fitting and predicting with our model.
    model.fit(trainX, trainY, context=train_context)

    # Note that context format adapts with the text.
    # For most tasks, the context for a sequence of text is a list of dictionaries.
    # For comparison and comparison_regressor, where the input X is a list of two text sequences, the context is also a list of two dictionary lists.
    # For multiple_choice, context must be given to both the question and answers. Specifically, for a given input, the context should be a list of n dictionary lists where the first corresponds to the question and the subsequent n-1 correspond to the answers.

    # See tests/test_auxiliary.py for examples.
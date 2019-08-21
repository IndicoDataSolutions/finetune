Dataset Loading
===============

Finetune supports providing input data as a list or as a data generator.  When a generator is provided as input, finetune
takes advantage of the :mod:`tf.data` module for data pipelining.


Providing text and targets in list format:

.. code-block:: python

    X = ['german shepherd', 'maine coon', 'persian', 'beagle']
    Y = ['dog', 'cat', 'cat', 'dog']
    model = Classifier()
    model.fit(X, Y)


Providing data as a generator:

.. code-block:: python

    df = pd.read_csv('pets.csv')

    # Even if raw data is greedily loaded,
    # using a generator allows us to defer data preprocessing
    def text_generator():
        for row in df.Text.values:
            yield row.Text

    # dataset_size must be specified if input is provided as generators
    model = Classifier(dataset_size=len(df))
    model.fit(text_generator)


.. figure:: https://i.imgur.com/kYL058E.png

.. module:: finetune


**Scikit-learn inspired model finetuning for natural language processing.**

:mod:`finetune` ships with a pre-trained language model from `"Improving Language Understanding by Generative Pre-Training" <https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf>`_
and builds off the `OpenAI/finetune-language-model repository <https://github.com/openai/finetune-transformer-lm>`_.

Source code for :mod:`finetune` is available `on github <https://github.com/IndicoDataSolutions/finetune>`_.


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


Installation
============
Finetune can be installed directly from PyPI by using `pip`

.. code-block:: bash

    pip install finetune


or installed directly from source:

.. code-block:: bash

    git clone https://github.com/IndicoDataSolutions/finetune
    cd finetune
    python3 setup.py develop
    python3 -m spacy download en

You can optionally run the provided test suite to ensure installation completed successfully.

.. code-block:: bash

    pip3 install pytest
    pytest


Docker
=======

If you'd prefer you can also run :mod:`finetune` in a docker container. The bash scripts provided assume you have a functional install of `docker <https://docs.docker.com/install>`_ and `nvidia-docker <https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>`_.

.. code-block:: bash

    # For usage with NVIDIA GPUs
    ./docker/build_gpu_docker.sh  # builds a docker image
    ./docker/start_gpu_docker.sh  # starts a docker container in the background
    docker exec -it finetune bash # starts a bash session in the docker container

For CPU-only usage:

.. code-block:: bash

    ./docker/build_cpu_docker.sh
    ./docker/start_cpu_docker.sh

Dataset Loading
===============

Finetune supports providing input data as a list or as a data generator.  When a generator is provided as input, finetune
takes advantage of the :mod:`tf.data` module for data pipelining


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

Prediction
==========

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


Saving and Loading Models
=========================

You can use the :py:func:`BaseModel.save()` and :py:func:`.load()` methods to serialize and deserialize trained models.

.. code-block:: python

    model = Classifier()
    model.fit(train_data, train_labels)
    model.save(filepath)


.. code-block:: python 

    model = Classifier.load(filepath)



Using Different Base Models (e.g. BERT, GPT2, RoBERTa)
=============================================

Finetune defaults to using OpenAI's GPT base model, but also supports other base model options.

.. code-block:: python
    
    from finetune.base_models import BERT, BERTLarge, GPT2, GPT2Medium, TextCNN, TCN, RoBERTa
    model = Classifier(base_model=RoBERTa)


Using the SequenceLabeler Class
============================================

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


Using Adapters and the DeploymentModel class
============================================

Alongside full finetuning, :mod:`finetune` also supports the adapter finetuning strategy from `"Parameter-Efficient Transfer Learning for NLP" <https://arxiv.org/abs/1902.00751>`_.
This dramatically shrinks the size of serialized model files.  When used in conjunction with the :class:`DeploymentModel` class at inference time, this enables quickly switching between target models.

.. code-block:: python

    # First we train and save a model using the adapter finetuning strategy
    from finetune import Classifier, DeploymentModel
    from finetune.base_models import GPT
    model = Classifier(adapter_size=64)
    model.fit(X, Y)
    model.save('adapter-model.jl')

    # Then we load it using the DeploymentModel wrapper
    deployment_model = DeploymentModel(featurizer=GPT)

    # Loading the featurizer only needs to be done once
    deployment_model.load_featurizer()

    # You can then cheaply load + predict with any adapter model that uses the
    # same base_model and adapter_size
    deployment_model.load_custom_model('adapter-model.jl')
    predictions = deployment_model.predict(testX)

    # Switching to another model takes only 2 seconds now rather than 20
    deployment_model.load_custom_model('another-adapter-model.jl')
    predictions = deployment_model.predict(testX) 

Using Auxiliary Info in Your Models
============================================

Our base models can also process arbitrary auxiliary information in addition to text, such as style (bolding, italics, etc.), semantics (part-of-speech tags, sentiment tags), or other forms,
as long as they describe specific spans of text.

.. code-block:: python

    # First we define the extra features we will be providing, as well as a default value that it will take if given data does not cover the text.
    # Auxiliary info can take the form of strings, booleans, floats, or ints.
    default = {'capitalized':False, 'part_of_speech':'unknown'}
    
    # Next we create context tags in a similar format to SequenceLabeling labels, as a list of lists of dictionaries:
    train_text = ['Intelligent process automation']
    train_context = [[
        {'text': 'Intelligent', 'capitalized': True, 'end': 11, 'start': 0, 'part_of_speech': 'ADJ'},
        {'text': 'process automation', 'capitalized': False, 'end': 30, 'start': 12, 'part_of_speech': 'NOUN'}, 
    ]]

    # Our input to the model is now a list containing the text, and then the context
    trainX = [train_text, train_context]

    # We indicate to the model that we are including auxiliary info by passing our default dictionary in with the kwarg default_context.
    model = Classifier(default_context=default)
    model.fit(trainX, trainY)



Code Examples
=============
For example usage of provided models, see the `finetune/datasets directory <https://github.com/IndicoDataSolutions/finetune/tree/master/finetune/datasets>`_.


Finetune API Reference
======================

.. autoclass:: finetune.Classifier
    :inherited-members:

.. autoclass:: finetune.Regressor
    :inherited-members:

.. autoclass:: finetune.MultiFieldClassifier
    :inherited-members:

.. autoclass:: finetune.MultiFieldRegressor
    :inherited-members:

.. autoclass:: finetune.MultiLabelClassifier
    :inherited-members:

.. autoclass:: finetune.SequenceLabeler
    :inherited-members:

.. autoclass:: finetune.Comparison
    :inherited-members:

.. autoclass:: finetune.OrdinalRegressor
    :inherited-members:

.. autoclass:: finetune.ComparisonOrdinalRegressor
    :inherited-members:

.. autoclass:: finetune.MultiTask
    :inherited-members:


Finetune Model Configuration Options
====================================

.. autoclass:: finetune.config.Settings
    :members:

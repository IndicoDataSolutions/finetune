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


Using Different Base Models (e.g. BERT, GPT2)
============================================

Finetune defaults to using OpenAI's GPT base model, but also supports other base model options.

.. code-block:: python
    
    from finetune.base_models import BERT, GPT2 TextCNN
    model = Classifier(base_model=BERT)


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

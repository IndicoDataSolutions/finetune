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

    ./docker/build_docker.sh      # builds a docker image
    ./docker/start_docker.sh      # starts a docker container in the background
    docker exec -it finetune bash # starts a bash session in the docker container


Code Examples
=============
For example usage of :class:`Classifier`, :class:`Entailment`, and :class:`SequenceLabeler`, see the `finetune/datasets directory <https://github.com/IndicoDataSolutions/finetune/tree/master/finetune/datasets>`_.


Finetune API Reference
======================
.. autoclass:: finetune.Classifier
    :inherited-members:

.. autoclass:: finetune.Entailment
    :inherited-members:

.. autoclass:: finetune.Model
    :inherited-members:

.. autoclass:: finetune.SequenceLabeler
    :inherited-members:

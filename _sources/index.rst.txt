.. figure:: https://i.imgur.com/kYL058E.png

.. module:: finetune

**Scikit-learn inspired model finetuning for natural language processing.**

Finetune is a library that allows users to leverage state-of-the-art pretrained NLP models for a wide variety of downstream tasks.

Finetune currently supports TensorFlow implementations of the following models:

1. **BERT**, from `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_.
2. **RoBERTa**, from `RoBERTa: A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_.
3. **GPT**, from `Improving Language Understanding by Generative Pre-Training <https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf>`_.
4. **GPT2**, from `Language Models are Unsupervised Multitask Learners <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_.
5. **TextCNN**, from `Convolutional Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>`_.
6. **Temporal Convolution Network**, from `An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling <https://arxiv.org/pdf/1803.01271.pdf>`_.


Huge thanks to Alec Radford and Jeff Wu for their hard work and quality research.
Source code for :mod:`finetune` is available `on github <https://github.com/IndicoDataSolutions/finetune>`_.

.. toctree::
    :maxdepth: 2
    :caption: General API

    installation
    basemodels
    quickstart
    datasetloading
    serializing
    config
    resource

.. toctree::
    :maxdepth: 2
    :caption: Special Features

    api
    cachedpredict
    chunk
    sequencelabeler
    adapter
    auxiliary


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


Code Examples
=============
For example usage of provided models, see the `finetune/datasets directory <https://github.com/IndicoDataSolutions/finetune/tree/master/finetune/datasets>`_.
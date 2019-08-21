Base Models (e.g. BERT, GPT2, RoBERTa)
======================================

Finetune defaults to using OpenAI's GPT base model, but also supports other base model options.

.. code-block:: python
    
    from finetune.base_models import BERT, BERTLarge, GPT2, GPT2Medium, TextCNN, TCN, RoBERTa

    # RoBERTa has provided state-of-the-art results on a variety of natural language tasks, as of late 2019
    model = Classifier(base_model=RoBERTa)

    # TextCNN and TCN are much faster convolutional models, at the expensive of downstream task performances
    fast_model = Classifier(base_model=TextCNN)
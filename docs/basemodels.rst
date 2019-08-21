Using Different Base Models (e.g. BERT, GPT2, RoBERTa)
======================================================

Finetune defaults to using OpenAI's GPT base model, but also supports other base model options.

.. code-block:: python
    
    from finetune.base_models import BERT, BERTLarge, GPT2, GPT2Medium, TextCNN, TCN, RoBERTa
    model = Classifier(base_model=RoBERTa)
Base Models (e.g. BERT, GPT2, RoBERTa, DistilBERT)
======================================

Finetune defaults to using OpenAI's GPT base model, but also supports other base model options.

.. code-block:: python
    
    from finetune.base_models import BERT, BERTLarge, GPT2, GPT2Medium, GPT2Large, TextCNN, TCN, RoBERTa, DistilBERT

    # RoBERTa has provided state-of-the-art results on a variety of natural language tasks, as of late 2019
    model = Classifier(base_model=RoBERTa)

    # The GPT and GPT2 model families allow experimentation with text generation
    model = LanguageModel(base_model=GPT2)

    # DistilBERT offers competetive finetuning performance with faster training and inference times thanks to its low parameter count
    model = Classifier(base_model=DistilBERT)

    # TextCNN and TCN are much faster convolutional models trained from scratch.  
    # They generally underperform their language model counterparts but may be appropriate if runtime is a concern
    fast_model = Classifier(base_model=TextCNN)
.. adapter:

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
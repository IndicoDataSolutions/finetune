<img src="https://i.imgur.com/kYL058E.png" height="150px">

**Scikit-learn style model finetuning for NLP**

`Finetune` ships with a pre-trained language model
from ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
and builds off the [OpenAI/finetune-language-model repository](https://github.com/openai/finetune-transformer-lm).

Finetune Quickstart Guide
=========================

Finetuning the base language model is as easy as calling `Classifier.fit`:

```python3
model = Classifier()               # Load base model
model.fit(trainX, trainY)          # Finetune base model on custom data
predictions = model.predict(testX) # [{'class_1': 0.23, 'class_2': 0.54, ..}, ..]
model.save(path)                   # Serialize the model to disk
```

Reload saved models from disk by using `LanguageModelClassifier.load`:

```
model = Classifier.load(path)
predictions = model.predict(testX)
```

Documentation
=============
Full documentation and an API Reference for `finetune` is available at [finetune.indico.io](https://finetune.indico.io).


Installation
============
Finetune can be installed directly from PyPI by using `pip`

```
pip3 install finetune
```

or installed directly from source:

```bash
git clone https://github.com/IndicoDataSolutions/finetune
cd finetune
python3 setup.py develop
python3 -m spacy download en
```

In order to run `finetune` on your host, you'll need a working copy of CUDA >= 8.0, libcudnn >= 6, tensorflow-gpu >= 1.6 and up to date nvidia-driver versions.

You can optionally run the provided test suite to ensure installation completed successfully.

```bash
pip3 install pytest
pytest
```


Docker
=======

If you'd prefer you can also run `finetune` in a docker container. The bash scripts provided assume you have a functional install of [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

```
./docker/build_docker.sh      # builds a docker image
./docker/start_docker.sh      # starts a docker container in the background
docker exec -it finetune bash # starts a bash session in the docker container
```

Code Examples
=============
For example usage of `Classifier`, `Entailment`, and `SequenceLabeler`, see the [finetune/datasets directory](https://github.com/IndicoDataSolutions/finetune/tree/master/finetune/datasets).  For purposes of simplicity and runtime these examples use smaller versions of the published datasets.

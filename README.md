<img src="https://i.imgur.com/kYL058E.png" width="100%">

**Scikit-learn style model finetuning for NLP**

`Finetune` ships with a pre-trained language model
from ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
and builds off the [OpenAI/finetune-language-model repository](https://github.com/openai/finetune-transformer-lm). Huge thanks to Alec Radford for his hard work and quality research.

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

If you have large amounts of unlabeled training data and only a small amount of labeled training data,
you can finetune in two steps for best performance.

```python3
model = Classifier()               # Load base model
model.fit(unlabeledX)              # Finetune base model on unlabeled training data
model.fit(trainX, trainY)          # Continue finetuning with a smaller amount of labeled data
predictions = model.predict(testX) # [{'class_1': 0.23, 'class_2': 0.54, ..}, ..]
model.save(path)                   # Serialize the model to disk
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
git clone -b master https://github.com/IndicoDataSolutions/finetune && cd finetune
python3 setup.py develop              # symlinks the git directory to your python path
pip3 install tensorflow-gpu --upgrade # or tensorflow-cpu
python3 -m spacy download en          # download spacy tokenizer
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
git clone https://github.com/IndicoDataSolutions/finetune && cd finetune
./docker/build_docker.sh      # builds a docker image
./docker/start_docker.sh      # starts a docker container in the background, forwards $PWD to /finetune
docker exec -it finetune bash # starts a bash session in the docker container
```


Model Types
============
`finetune` ships with a half dozen different classes for finetuning the base language model on different task types.

- `Classifier`
- `Regressor`
- `SequenceLabeler`
- `MultifieldClassifier`
- `MultifieldRegressor`

For example usage of each of these model types, see the [finetune/datasets directory](https://github.com/IndicoDataSolutions/finetune/tree/master/finetune/datasets).
For purposes of simplicity and runtime these examples use smaller versions of the published datasets.

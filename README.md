<img src="https://i.imgur.com/kYL058E.png" width="100%">

**Scikit-learn style model finetuning for NLP**

Finetune is a library that allows users to leverage state-of-the-art pretrained NLP models for a wide variety of downstream tasks.

Finetune currently supports TensorFlow implementations of the following models:

1. **BERT**, from ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805)
2. **RoBERTa**, from ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692)
3. **GPT**, from ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
4. **GPT2**, from ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
5. **TextCNN**, from ["Convolutional Neural Networks for Sentence Classification"](https://arxiv.org/abs/1408.5882)
6. **Temporal Convolution Network**, from ["An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"](https://arxiv.org/pdf/1803.01271.pdf)
7. **DistilBERT** from ["Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT"](https://medium.com/huggingface/distilbert-8cf3380435b5)


| Section | Description |
|-|-|
| [API Tour](#finetune-api-tour) | Base models, configurables, and more |
| [Installation](#installation-tour) | How to install using pip or directly from source |
| [Finetune with Docker](#docker) | Finetune and inference within a Docker Container |
| [Documentation](https://finetune.indico.io/) | Full API documentation |

# Finetune API Tour

Finetuning the base language model is as easy as calling `Classifier.fit`:

```python3
model = Classifier()               # Load base model
model.fit(trainX, trainY)          # Finetune base model on custom data
model.save(path)                   # Serialize the model to disk
...
model = Classifier.load(path)      # Reload models from disk at any time
predictions = model.predict(testX) # [{'class_1': 0.23, 'class_2': 0.54, ..}, ..]
```

Choose your desired base model from `finetune.base_models`:
```python3
from finetune.base_models import BERT, RoBERTa, GPT, GPT2, TextCNN, TCN
model = Classifier(base_model=BERT)
```

Optimize your model with a variety of configurables. A detailed list of all config items can be found [in the finetune docs](https://finetune.indico.io/config.html).
```python3
model = Classifier(low_memory_mode=True, lr_schedule="warmup_linear", max_length=512, l2_reg=0.01, oversample=True, ...)
```

The library supports finetuning for a number of tasks. A detailed description of all target models can be found [in the finetune API reference](https://finetune.indico.io/api.html).
```python3
from finetune import *
models = (Classifier, MultiLabelClassifier, MultiFieldClassifier, MultipleChoice, # Classify one or more inputs into one or more classes
          Regressor, OrdinalRegressor, MultifieldRegressor,                       # Regress on one or more inputs
          SequenceLabeler, Association,                                           # Extract tokens from a given class, or infer relationships between them
          Comparison, ComparisonRegressor, ComparisonOrdinalRegressor,            # Compare two documents for a given task
          LanguageModel, MultiTask,                                               # Further pretrain your base models
          DeploymentModel                                                         # Wrapper to optimize your serialized models for a production environment
          )
```
For example usage of each of these target types, see the [finetune/datasets directory](https://github.com/IndicoDataSolutions/finetune/tree/master/finetune/datasets).
For purposes of simplicity and runtime these examples use smaller versions of the published datasets.






If you have large amounts of unlabeled training data and only a small amount of labeled training data,
you can finetune in two steps for best performance.

```python3
model = Classifier()               # Load base model
model.fit(unlabeledX)              # Finetune base model on unlabeled training data
model.fit(trainX, trainY)          # Continue finetuning with a smaller amount of labeled data
predictions = model.predict(testX) # [{'class_1': 0.23, 'class_2': 0.54, ..}, ..]
model.save(path)                   # Serialize the model to disk
```

# Installation

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

In order to run `finetune` on your host, you'll need a working copy of tensorflow-gpu >= 1.14.0 and up to date nvidia-driver versions.

You can optionally run the provided test suite to ensure installation completed successfully.

```bash
pip3 install pytest
pytest
```


# Docker

If you'd prefer you can also run `finetune` in a docker container. The bash scripts provided assume you have a functional install of [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

```
git clone https://github.com/IndicoDataSolutions/finetune && cd finetune

# For usage with NVIDIA GPUs
./docker/build_gpu_docker.sh      # builds a docker image
./docker/start_gpu_docker.sh      # starts a docker container in the background, forwards $PWD to /finetune

docker exec -it finetune bash # starts a bash session in the docker container
```

For CPU-only usage:
```
./docker/build_cpu_docker.sh
./docker/start_cpu_docker.sh
```

# Documentation
Full documentation and an API Reference for `finetune` is available at [finetune.indico.io](https://finetune.indico.io).


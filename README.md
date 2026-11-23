[![DOI](https://zenodo.org/badge/137103159.svg)](https://zenodo.org/badge/latestdoi/137103159)

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
| [Installation](#installation) | Build with Docker or install from source |
| [Finetune with Docker](#docker) | Build a container from this repository |
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

Build from this repository with Docker (recommended), or install from source.
Use Python 3.12 for a source install to match the Dockerfile.
The current encoder API requires `tokenizers<0.23`; the commands below constrain that dependency.

Clone the maintained development branch:

```bash
git clone --branch development https://github.com/IndicoDataSolutions/finetune.git
cd finetune
```

## Docker

Install [Docker](https://docs.docker.com/engine/install/), then build the provided Dockerfile:

```bash
docker build -t finetune:source .
docker run --rm -it -v "$PWD:/Finetune" finetune:source bash
```

Inside the container, install the mounted source and the spaCy tokenizer:

```bash
python -m pip install -e '.[test]' 'tokenizers<0.23'
python -m spacy download en_core_web_sm
python -c "import finetune"
```

For NVIDIA GPU access, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and add `--gpus all` to the `docker run` command.
The Dockerfile includes TensorFlow 2.20.0 and its CUDA dependencies.

## Install from source

From the cloned repository, create a virtual environment and install the source with TensorFlow:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[tf,test]' 'tokenizers<0.23'
python -m spacy download en_core_web_sm
python -c "import finetune"
```

To run the test suite in either environment:

```bash
python -m pytest
```

# Documentation
Full documentation and an API Reference for `finetune` is available at [finetune.indico.io](https://finetune.indico.io).


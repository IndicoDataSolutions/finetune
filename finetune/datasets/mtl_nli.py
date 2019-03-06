import os
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from finetune import MultiTask, Classifier, MultiFieldClassifier, Comparison, MultiFieldRegressor
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank
from finetune.datasets.quora_similarity import QuoraDuplicate
from finetune.datasets.reuters import Reuters
logging.basicConfig(level=logging.DEBUG)

DATA_PATH = "/root/code/Data/glue_data"

def COLA_train(data_folder):
    col_names_train = ["id", "target", "na2", "text"]
    data_folder += "/CoLA"

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", names=col_names_train)
    train_X, train_Y = train_dataframe.text.values, train_dataframe.target.values

    return "cola", train_X, train_Y, Classifier

def SST_train(data_folder):
    data_folder += "/SST-2"
    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t",quoting=3)
    
    train_X, train_Y = train_dataframe.sentence.values, train_dataframe.label.values

    return "sst", train_X, train_Y, Classifier

def MRPC_train(data_folder):
    data_folder += "/MRPC"

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3).astype(str)
    train_X, train_Y = list(zip(train_dataframe["#1 String"].values,train_dataframe["#2 String"].values)), train_dataframe.Quality.values

    return "mrpc", train_X, train_Y, Comparison

def STS_train(data_folder):
    data_folder += "/STS-B"

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)
    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.score.values

    return "sts-b", train_X, train_Y, MultiFieldRegressor

def QQP_train(data_folder):
    data_folder += "/QQP"

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3).astype(str)

    train_X, train_Y = list(zip(train_dataframe.question1.values,train_dataframe.question2.values)), train_dataframe.is_duplicate.values
    return "qqp", train_X, train_Y, Comparison


def MNLI_train(data_folder):
    data_folder += "/MNLI"

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3).astype(str)
    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.gold_label.values
    return "mnli", train_X, train_Y, MultiFieldClassifier



def QNLI_train(data_folder):
    data_folder += "/QNLI"
    
    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3).astype(str)
    train_X, train_Y = list(zip(train_dataframe.question.values,train_dataframe.sentence.values)), train_dataframe.label.values
    return "qnli", train_X, train_Y, MultiFieldClassifier
    


def RTE_train(data_folder):
    data_folder += "/RTE"
    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)

    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.label.values
    return "rte", train_X, train_Y, MultiFieldClassifier

def WNLI_train(data_folder, output_folder, gpu_num):
    data_folder += "/WNLI"

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)

    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.label.values
    return "wnli", train_X, train_Y, MultiFieldClassifier

if __name__ == "__main__":
    # Train and evaluate on SST

    tasks = {}
    Xs = {}
    Ys = {}
    for fn in [
#            WNLI_train,
            RTE_train,
            QNLI_train,
            MNLI_train,
            SST_train,
            COLA_train,
            MRPC_train,
#            STS_train,
            QQP_train,
        ]:
        name, x, y, cls = fn(DATA_PATH)
        tasks[name] = cls
        Xs[name] = x[:100000]
        Ys[name] = y[:100000]
        
    
    model = MultiTask(tasks=tasks, debugging_logs=True, lr=6.25e-6, n_epochs=30, tensorboard_folder="mtl_glue_1k", max_grad_norm=10.0)

    model.fit(Xs, Ys)
    model.create_base_model("glue_base_1k.jl", exists_ok=True)


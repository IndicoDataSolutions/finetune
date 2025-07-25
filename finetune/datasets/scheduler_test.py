import codecs
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

import joblib as jl
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from sklearn.model_selection import train_test_split

from finetune import SequenceLabeler
from finetune.base_models import GPT, GPT2, TCN, RoBERTa
from finetune.datasets.reuters import Reuters
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.scheduler import Scheduler
from finetune.util.metrics import annotation_report

if __name__ == "__main__":
    dataset = Reuters().dataframe
    dataset["annotations"] = [
        json.loads(annotation) for annotation in dataset["annotations"]
    ]
    trainX, testX, trainY, testY = train_test_split(
        dataset.texts.values, dataset.annotations.values, test_size=0.2, random_state=42
    )

    n_models = 40
    #    fn_to_preds = dict()
    #    for i in range(n_models):
    #        fn = "scheduler_models/scheduler_{}.model".format(i)
    #        model = SequenceLabeler(base_model=RoBERTa)
    #        model.fit([trainX[i]], [trainY[i]])
    #        model.save(fn)
    #        model = SequenceLabeler.load(fn)
    #        preds = model.predict(testX)
    #        fn_to_preds[fn] = preds
    #    jl.dump(fn_to_preds, "fn_to_preds")
    fn_to_preds = jl.load("fn_to_preds")
    fns = list(fn_to_preds.keys())
    sched = Scheduler()

    while True:
        model = fns[random.randint(0, n_models - 1)]
        start = time.time()
        preds = sched.predict(model, testX)
        assert preds == fn_to_preds[model]
        print(
            "Elapsed Time: {}, Loaded Models: {}".format(
                time.time() - start, len(sched.loaded_models)
            )
        )

import pandas as pd
import os
import json
import time
from finetune.base_models import RoBERTa, ModernBert, ModernBertLarge, BERTLarge
from finetune import SequenceLabeler

from sequence_metrics import metrics

MODELS = {
    "modern_bert": ModernBert,
    "modern_bert_large": ModernBertLarge,
    "roberta": RoBERTa,
    "bert_large": BERTLarge,
}

def get_model(model_name):
    return SequenceLabeler(
        base_model=MODELS[model_name],
        auto_negative_sampling=True,
        low_memory_mode=True,
        class_weights="sqrt",
        collapse_whitespace=True,
    )


def get_dataset_split(dataset_path, split_name):
    df = pd.read_csv(os.path.join(dataset_path, f"{split_name}.csv"))
    x = []
    y = []
    for i, row in df.iterrows():
        x.append(row["text"])
        y.append(json.loads(row["labels"]))
    return x, y


def train_model(model_name, dataset_path):
    model = get_model(model_name)
    x, y = get_dataset_split(dataset_path, "train")
    model.fit(x, y)
    return model

def evaluate_model(model: SequenceLabeler, dataset_path):
    x, y = get_dataset_split(dataset_path, "test")
    preds = model.predict(x)
    return metrics.get_all_metrics(preds, y)

if __name__ == "__main__":
    for dataset in ["/datasets/charities", "/datasets/fcc_invoices", "/datasets/nda"]:
        print(f"Evaluating {dataset}")
        for model_name in ["modern_bert", "roberta"]: #, "modern_bert_large", "bert_large"]:
            results_path = f"results/{dataset.split('/')[-1]}_{model_name}.json"
            if os.path.exists(results_path):
                print(f"Skipping {model_name} for {dataset} because results already exist")
                continue
            print(f"Model: {model_name}")
            start_time = time.time()
            model = train_model(model_name, dataset)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training time: {training_time} seconds")
            start_time = time.time()
            results = evaluate_model(model, dataset)
            end_time = time.time()
            evaluation_time = end_time - start_time
            print(f"Evaluation time: {evaluation_time} seconds")
            results["training_time"] = training_time
            results["evaluation_time"] = evaluation_time
            with open(results_path, "w") as f:
                json.dump(results, f, indent=1)

import json
import os
import time

import pandas as pd
from sequence_metrics import metrics

from finetune import SequenceLabeler
from finetune.base_models import BERTLarge, ModernBert, ModernBertLarge, RoBERTa

MODELS = {
    "modern_bert": ModernBert,
    "modern_bert_large": ModernBertLarge,
    "roberta": RoBERTa,
    "bert_large": BERTLarge,
}


def get_model(model_name, **model_kwargs):
    return SequenceLabeler(
        base_model=MODELS[model_name],
        auto_negative_sampling=True,
        low_memory_mode=True,
        class_weights="sqrt",
        collapse_whitespace=True,
        **model_kwargs,
    )


def get_dataset_split(dataset_path, split_name):
    df = pd.read_csv(os.path.join(dataset_path, f"{split_name}.csv"))
    x = []
    y = []
    for i, row in df.iterrows():
        x.append(row["text"])
        y.append(json.loads(row["labels"]))
    return x, y


def train_model(model_name, dataset_path, **model_kwargs):
    model = get_model(model_name, **model_kwargs)
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
        for model_name, version in [
            ("modern_bert", 4),
            ("roberta", 3),
        ]:  # , "modern_bert_large", "bert_large"]:
            for optimize_for in ["predict_speed", "accuracy"]:
                results_path = f"results/{dataset.split('/')[-1]}_{model_name}_{optimize_for}_v{version}.json"
                if model_name == "roberta" and optimize_for == "accuracy":
                    # This has already been run.
                    continue
                if os.path.exists(results_path):
                    print(
                        f"Skipping {model_name} for {dataset} because results already exist"
                    )
                    continue
                print(f"Model: {model_name}")
                start_time = time.time()
                model = train_model(model_name, dataset, optimize_for=optimize_for)
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

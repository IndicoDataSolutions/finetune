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

def get_model(model_name, **model_kwargs):
    return SequenceLabeler(
        base_model=MODELS[model_name],
        auto_negative_sampling=True,
        low_memory_mode=True,
        class_weights="sqrt",
        collapse_whitespace=True,
        **model_kwargs
    )


def get_dataset_split(dataset_path, split_name):
    df = pd.read_csv(os.path.join(dataset_path, f"{split_name}.csv"))
    x = []
    y = []
    for i, row in df.iterrows():
        x.append(row["text"])
        y.append(json.loads(row["labels"]))
    return x, y


def train_model(model_name, dataset_path, max_data=None, **model_kwargs):
    model = get_model(model_name, **model_kwargs)
    x, y = get_dataset_split(dataset_path, "train")
    if max_data is not None:
        x = x[:max_data]
        y = y[:max_data]
    model.fit(x, y)
    return model

def evaluate_model(model: SequenceLabeler, dataset_path):
    x, y = get_dataset_split(dataset_path, "test")
    preds = model.predict(x)
    return metrics.get_all_metrics(preds, y)

if __name__ == "__main__":
    for dataset in ["/datasets/charities"]:
        results = {"predict_speed": {}, "accuracy": {}}
        results_path = f"{dataset}_accuracy.json"
        for optimize_for in ["accuracy"]:
            start_time = time.time()
            model = train_model("modern_bert", dataset, optimize_for=optimize_for, n_epochs=1, max_data=5)
            x, y = get_dataset_split(dataset, "test")
            for predict_batch_size in range(1, 100):
                model.config.predict_batch_size = predict_batch_size
                start = time.time()
                model.predict(x)
                end = time.time()
                print(f"Predict time: {end - start} seconds")
                results[optimize_for][predict_batch_size] = end - start
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=1)
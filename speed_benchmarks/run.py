import time

from tabulate import tabulate
from finetune import Classifier, SequenceLabeler
from finetune.base_models import RoBERTa
from synthetic_data import multi_label_sequence_data, sequence_data, classification_data

def benchmark(model_cls, config, x, y, runs):
    train_time = 0
    inference_time = 0	
    for _ in range(runs):
        model = model_cls(**config)
        train_start_time = time.time()
        model.fit(x, y)
        train_time += time.time() - train_start_time	

        infer_start_time = time.time()
        model.predict(x)
        inference_time += time.time() - infer_start_time
    return train_time / runs, inference_time / runs

def benchmark_classification(config, runs=5):
    x, y = classification_data()
    return benchmark(Classifier, config, x, y, runs=runs)

def benchmark_sequence(config, runs=5):
    x, y = sequence_data()
    return benchmark(SequenceLabeler, config, x, y, runs=runs)

def benchmark_multi_sequence(config, runs=5):
    x, y = multi_label_sequence_data()
    config["multi_label_sequences"] = True
    return benchmark(SequenceLabeler, config, x, y, runs=runs)

if __name__ == "__main__":
    average_over_n_runs = 1
    base_config = {"base_model": RoBERTa}
    output = []
    headers = ["Model", "Optimized For", "Train Time", "Predict Time"]
    for optimize_for in ["speed", "accuracy", "predict_speed"]:
        config = dict(base_config)
        config["optimize_for"] = optimize_for
        clf_train, clf_infer = benchmark_classification(config, runs=average_over_n_runs)
        output.append(["Classification", optimize_for, clf_train, clf_infer])

        seq_train, seq_infer = benchmark_sequence(config, runs=average_over_n_runs)
        output.append(["Sequence", optimize_for, seq_train, seq_infer])

        multi_seq_train, multi_seq_infer = benchmark_multi_sequence(config, runs=average_over_n_runs)
        output.append(["Multi Sequence", optimize_for, multi_seq_train, multi_seq_infer])
    print(tabulate(output, headers=headers))
    

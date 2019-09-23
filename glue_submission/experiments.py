import traceback
import os

from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, f1_score
from finetune import Classifier, Comparison, MultiFieldRegressor, MultiFieldClassifier
from finetune.base_models import GPCModel
import pandas as pd

import numpy as np

BPE_LEN_FACTOR = 1.5
MAX_LEN = 512

DATA_PATH = "~/code/Data/glue_data/"
SUBMISSION_PATH = "~/code/Data/glue_data/submission"

BASE_MODEL=GPCModel

def estimate_max_len(train_X):
    lens = []
    for sample in train_X:
        if type(sample) != str:
            s = " ".join(sample)
        else:
            s = sample
        lens.append(len(s.split(" ")) * BPE_LEN_FACTOR)
    return int(min(MAX_LEN, max(lens)))


def COLA_train(data_folder, output_folder, gpu_num):
    col_names_train = ["id", "target", "na2", "text"]
    data_folder += "/CoLA"

    output_file = os.path.join(output_folder, "CoLA.tsv")

    if os.path.exists(output_file):
        return


    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", names=col_names_train)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t", names=col_names_train)
    dev_X, dev_Y = dev_dataframe.text.values, dev_dataframe.target.values
    train_X, train_Y = train_dataframe.text.values, train_dataframe.target.values
    max_len = estimate_max_len(train_X)
    model = Classifier(
        base_model=BASE_MODEL,
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        max_length=max_len
    )
    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t")
    test_i = test_dataframe.index.values
    test_text = test_dataframe.sentence.values
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)

    #Eval
    dev_pred = model.predict(dev_X)
    print("\n\n\nCola Matthews Corr: {}\n\n\n".format(matthews_corrcoef(dev_Y, dev_pred)))

    model.save("cola_glue.jl")

def SST_train(data_folder, output_folder, gpu_num):
    data_folder += "/SST-2"
    output_file = os.path.join(output_folder, "SST-2.tsv")
    if os.path.exists(output_file):
        return

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t",quoting=3)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t",quoting=3)

    dev_X, dev_Y = dev_dataframe.sentence.values, dev_dataframe.label.values
    train_X, train_Y = train_dataframe.sentence.values, train_dataframe.label.values

    model = Classifier(
        base_model=BASE_MODEL,
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        max_length=estimate_max_len(train_X)
    )

    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t",quoting=3)
    test_i = test_dataframe.index.values
    test_text = test_dataframe.sentence.values
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)

    #Eval
    dev_pred = model.predict(dev_X)
    print("\n\n\nSST-2 accuracy: {}\n\n\n".format(np.mean(dev_Y == dev_pred)))

    model.save("sst_glue.jl")


def MRPC_train(data_folder, output_folder, gpu_num):
    data_folder += "/MRPC"
    output_file = os.path.join(output_folder, "MRPC.tsv")
    if os.path.exists(output_file):
        return

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t", quoting=3)

    dev_X, dev_Y = list(zip(dev_dataframe["#1 String"].values, dev_dataframe["#2 String"].values)), dev_dataframe.Quality.values

    train_X, train_Y = list(zip(train_dataframe["#1 String"].values,train_dataframe["#2 String"].values)), train_dataframe.Quality.values
    for x in train_X:
        if type(x[0]) != str or type(x[1]) != str:
            print(x)
    model = Comparison(
        base_model=BASE_MODEL
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        max_length=estimate_max_len(train_X)
    )

    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t", quoting=3)
    test_i = test_dataframe.index.values
    test_text = list(zip(test_dataframe["#1 String"].values, test_dataframe["#2 String"].values))
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)

    #Eval
    dev_pred = model.predict(dev_X)
    print("\n\n\nMRPC accuracy: {}\n\n\n".format(np.mean(dev_Y == dev_pred)))
    print("\n\n\nMRPC F1: {}\n\n\n".format(f1_score(dev_Y, dev_pred)))
    model.save("MRPC_glue.jl")

def STS_train(data_folder, output_folder, gpu_num):
    data_folder += "/STS-B"
    output_file = os.path.join(output_folder, "STS-B.tsv")
    if os.path.exists(output_file):
        return
    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t", quoting=3)

    dev_X, dev_Y = list(zip(dev_dataframe.sentence1.values, dev_dataframe.sentence2.values)), dev_dataframe.score.values

    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.score.values

    model = MultiFieldRegressor(
        base_model=BASE_MODEL,
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        regression_loss="L1",
        max_length=estimate_max_len(train_X)
    )
    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t", quoting=3)
    test_i = test_dataframe.index.values
    test_text = list(zip(test_dataframe.sentence1.values, test_dataframe.sentence2.values))
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)
    #Eval

    dev_pred = model.predict(dev_X)
    print("\n\n\nSTS Pearson Rank: {}\n\n\n".format(pearsonr(dev_Y, dev_pred)))
    model.save("STS_glue.jl")

def QQP_train(data_folder, output_folder, gpu_num):
    data_folder += "/QQP"
    output_file = os.path.join(output_folder, "QQP.tsv")

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3).astype(str)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t", quoting=3).astype(str)

    dev_X, dev_Y = list(zip(dev_dataframe.question1.values, dev_dataframe.question2.values)), dev_dataframe.is_duplicate.values

    train_X, train_Y = list(zip(train_dataframe.question1.values,train_dataframe.question2.values)), train_dataframe.is_duplicate.values

    model = Comparison(
        base_model=BASE_MODEL,
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        max_length=estimate_max_len(train_X)
    )

    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t", quoting=3)
    test_i = test_dataframe.index.values
    test_text = list(zip(test_dataframe.question1.values, test_dataframe.question2.values))
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)
    #Eval
    dev_pred = model.predict(dev_X)
    print("\n\n\nQQP accuracy: {}\n\n\n".format(np.mean(dev_Y == dev_pred)))
#    print("\n\n\nQQP F1: {}\n\n\n".format(f1_score(dev_Y, dev_pred)))
    model.save("QQP_glue.jl")


def MNLI_train(data_folder, output_folder, gpu_num):
    print("Running MNLI")
    data_folder += "/MNLI"
    output_file_matched = os.path.join(output_folder, "MNLI-m.tsv")
    output_file_mismatched = os.path.join(output_folder, "MNLI-mm.tsv")
    output_file_diagnostic = os.path.join(output_folder, "AX.tsv")
#    if os.path.exists(output_file_diagnostic):
#        return

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3).astype(str)
    dev_dataframe_matched = pd.read_csv(os.path.join(data_folder, "dev_matched.tsv"), sep="\t", quoting=3).astype(str)
    dev_dataframe_mismatched = pd.read_csv(os.path.join(data_folder, "dev_mismatched.tsv"), sep="\t", quoting=3).astype(str)

    dev_dataframe = pd.concat([dev_dataframe_matched, dev_dataframe_mismatched])

    dev_X, dev_Y = list(zip(dev_dataframe.sentence1.values, dev_dataframe.sentence2.values)), dev_dataframe.gold_label.values
    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.gold_label.values

    model = MultiFieldClassifier(
        base_model=BASE_MODEL,
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        max_length=estimate_max_len(train_X)
    )

    model.fit(train_X, train_Y)

    test_dataframe_matched = pd.read_csv(os.path.join(data_folder, "test_matched.tsv"), sep="\t", quoting=3).astype(str)
    test_dataframe_mismatched = pd.read_csv(os.path.join(data_folder, "test_mismatched.tsv"), sep="\t", quoting=3).astype(str)
    test_dataframe_diagnostic = pd.read_csv(os.path.join(data_folder, "../diagnostic/diagnostic.tsv"), sep="\t", quoting=3).astype(str)
    model.save("MNLI_glue.jl")

    for test_dataframe, output_file in [
            (test_dataframe_matched, output_file_matched),
            (test_dataframe_mismatched, output_file_mismatched),
            (test_dataframe_diagnostic, output_file_diagnostic)
    ]:
        test_i = test_dataframe.index.values
        test_text = list(zip(test_dataframe.sentence1.values, test_dataframe.sentence2.values))
        test_pred = model.predict(test_text)
        pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)

    #Eval                                                                                                                                                                           \

    dev_pred = model.predict(dev_X)
    print("\n\n\nMNLI accuracy: {}\n\n\n".format(np.mean(dev_Y == dev_pred)))
    model.save("MNLI_glue.jl")



def QNLI_train(data_folder, output_folder, gpu_num):
    data_folder += "/QNLI"
    output_file = os.path.join(output_folder, "QNLI.tsv")
    if os.path.exists(output_file):
        return

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t", quoting=3)

    dev_X, dev_Y = list(zip(dev_dataframe.question.values, dev_dataframe.sentence.values)), dev_dataframe.label.values

    train_X, train_Y = list(zip(train_dataframe.question.values,train_dataframe.sentence.values)), train_dataframe.label.values
    model = MultiFieldClassifier(
        base_model=BASE_MODEL,
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        max_length=estimate_max_len(train_X)
    )

    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t", quoting=3)
    test_i = test_dataframe.index.values
    test_text = list(zip(test_dataframe.question.values, test_dataframe.sentence.values))
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)

    #Eval

    dev_pred = model.predict(dev_X)
    print("\n\n\nQNLI accuracy: {}\n\n\n".format(np.mean(dev_Y == dev_pred)))
    model.save("QNLI_glue.jl")

def RTE_train(data_folder, output_folder, gpu_num):
    data_folder += "/RTE"
    output_file = os.path.join(output_folder, "RTE.tsv")
    if os.path.exists(output_file):
        return

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t", quoting=3)

    dev_X, dev_Y = list(zip(dev_dataframe.sentence1.values, dev_dataframe.sentence2.values)), dev_dataframe.label.values

    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.label.values
    model = MultiFieldClassifier(
        val_set=(dev_X, dev_Y), visible_gpus=[gpu_num], max_length=estimate_max_len(train_X), base_model=BASE_MODEL,
    )

    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t", quoting=3)
    test_i = test_dataframe.index.values
    test_text = list(zip(test_dataframe.sentence1.values, test_dataframe.sentence2.values))
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)

    #Eval

    dev_pred = model.predict(dev_X)
    print("\n\n\nRTE accuracy: {}\n\n\n".format(np.mean(dev_Y == dev_pred)))
    model.save("RTE_glue.jl")

def WNLI_train(data_folder, output_folder, gpu_num):
    data_folder += "/WNLI"
    output_file = os.path.join(output_folder, "WNLI.tsv")
    if os.path.exists(output_file):
        return

    train_dataframe = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", quoting=3)
    dev_dataframe = pd.read_csv(os.path.join(data_folder, "dev.tsv"), sep="\t", quoting=3)

    dev_X, dev_Y = list(zip(dev_dataframe.sentence1.values, dev_dataframe.sentence2.values)), dev_dataframe.label.values

    train_X, train_Y = list(zip(train_dataframe.sentence1.values,train_dataframe.sentence2.values)), train_dataframe.label.values
    model = MultiFieldClassifier(
        base_model=BASE_MODEL,
        val_set=(dev_X, dev_Y),
        visible_gpus=[gpu_num],
        max_length=estimate_max_len(train_X)
    )

    model.fit(train_X, train_Y)

    test_dataframe = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t", quoting=3)
    test_i = test_dataframe.index.values
    test_text = list(zip(test_dataframe.sentence1.values, test_dataframe.sentence2.values))
    test_pred = model.predict(test_text)
    pd.DataFrame(dict(id=test_i, label=test_pred)).to_csv(output_file, sep="\t", index=False)

    #Eval

    dev_pred = model.predict(dev_X)
    print("\n\n\nWNLI accuracy: {}\n\n\n".format(np.mean(dev_Y == dev_pred)))
    model.save("WNLI_glue.jl")



    gpu_map = dict(enumerate(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    print(gpu_map)
    pool_id = multiprocessing.current_process()._identity[0]
    print(pool_id)
    try:
        gpu_id = gpu_map[pool_id]
        print(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        fn(DATA_PATH, SUBMISSION_PATH, int(gpu_id))
        print("fn finished")
    except Exception as e:
        print("Exception occoured:")
        print(traceback.format_exc())


if __name__=="__main__":
    import multiprocessing
    pool = multiprocessing.Pool(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    for _ in pool.imap_unordered(
        run,
        [
#            WNLI_train,
#            RTE_train,
            QNLI_train,
#            MNLI_train
#            SST_train,
            COLA_train,
#            MRPC_train,
#            STS_train,
#            QQP_train,
        ]
    ):
        pass


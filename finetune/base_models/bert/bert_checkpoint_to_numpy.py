import tensorflow as tf
import joblib  as jl

if __name__ == "__main__":
    PATTERN = "/Users/work/Downloads/{}/bert_model.ckpt"
    OUTPUT_NAME = "/Users/work/git/finetune-transformer-lm/finetune/model/bert/bert_{}.jl"

    models = [
        ("multi_cased_L-12_H-768_A-12", "small_multi_cased"),
        ("cased_L-12_H-768_A-12", "small_cased"),
        ("cased_L-24_H-1024_A-16", "large_cased")

    ]
    for folder, out in models:
        reader = tf.train.NewCheckpointReader(PATTERN.format(folder))
        keys = list(reader.get_variable_to_shape_map().keys())
        output_dict = {'model/featurizer/' + k + ":0": reader.get_tensor(k) for k in keys}
        jl.dump(output_dict, OUTPUT_NAME.format(out))

import tensorflow as tf
import joblib  as jl

if __name__ == "__main__":
    PATTERN = "/Users/work/Downloads/multi_cased_L-12_H-768_A-12/bert_model.ckpt"
    OUTPUT_NAME = "/Users/work/git/finetune-transformer-lm/finetune/model/bert/bert_small_multi_cased.jl"
    reader = tf.train.NewCheckpointReader(PATTERN)
    keys = list(reader.get_variable_to_shape_map().keys())
    output_dict = {'model/featurizer/' + k + ":0": reader.get_tensor(k) for k in keys}
    jl.dump(output_dict, OUTPUT_NAME)


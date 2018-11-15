"""
This script is used to install a normal finetune save file to the base model format for further finetuning.

Example usage:
```
python3 save_file_to_base_model.py --input_path=save_model.jl --output_name=wikipedia_model.jl
```
"""
import os

import joblib as jl
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", None, "Path to the saved model to be converted")
flags.DEFINE_string("output_name", None, "Filename for the newly created base model file")


def main(_):
    if FLAGS.input_path is None or FLAGS.output_name is None:
        print("Must provide input_path and output_name arguments")
        exit(1)

    weights, _ = jl.load(FLAGS.input_path)
    weights_stripped = {k: v for k, v in weights.items() if "featurizer" in k and "OptimizeLoss" not in k}
    base_model_path = os.path.join(os.path.dirname(__file__), "model", FLAGS.output_name)
    jl.dump(weights_stripped, base_model_path)
    print("Complete!")

if __name__ == "__main__":
    tf.app.run()

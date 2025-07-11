import os
from concurrent.futures import ThreadPoolExecutor
import logging
import re

import joblib
import numpy as np
import tensorflow as tf

from finetune.config import get_config

LOGGER = logging.getLogger("finetune")


def should_be_randomly_initialized(name):
    return "OptimizeLoss" in name or "global_step" in name


def set_weights(model: tf.keras.Model, weights: dict[str, np.ndarray], all_vars: dict[str, tf.Variable]):
    for name, np_weight in weights.items():
        if name.endswith("we:0") or "bert/embeddings/position_embedding" in name:
            np_weight = np_weight[:all_vars[name].shape[0]]
        all_vars[name].assign(np_weight)

def own_variables(layer):
    """Return weights that belong *directly* to `layer`."""
    children_weights = []
    for child in layer._layers:
        children_weights.extend(child.weights)
    return [v for v in layer.weights if not any(v is child_weight for child_weight in children_weights)]

def get_fully_qualified_variable_paths(root: tf.keras.layers.Layer) -> dict[str, tf.Variable]:
    root_name = root.name
    results = {}
    for variable in own_variables(root):
        results[f"{root_name}/{variable.name}:0"] = variable
    for layer in root._layers:
        for k, v in get_fully_qualified_variable_paths(layer).items():
            results[f"{root_name}/{k}"] = v
    return results

class Saver:
    def __init__(
        self,
        fallback_filename=None,
        variable_transforms=None,
        save_dtype=None,
        permit_uninitialized=None,
    ):
        self.variable_transforms = variable_transforms or []
        self.save_dtype = save_dtype
        if fallback_filename is not None:
            self.set_fallback(fallback_filename)
        self.permit_uninitialized = permit_uninitialized
        self.variables = None

    def set_fallback(self, fallback_filename):
        self.tpe = ThreadPoolExecutor()
        if not os.path.exists(fallback_filename):
            raise FileNotFoundError("Error loading base model {} - file not found.".format(fallback_filename))
        self.fallback_filename = fallback_filename
        self.fallback_future = self.tpe.submit(joblib.load, fallback_filename)
        self.fallback_ = None

    @property
    def fallback(self):
        if self.fallback_ is None:
            self.fallback_ = self.fallback_future.result()
            self.fallback_future = None
            self.tpe.shutdown()
        return self.fallback_

    def save_model(self, model: tf.keras.Model, finetune_obj, path):
        variables = get_fully_qualified_variable_paths(model)
        if isinstance(path, str):
            folder = os.path.dirname(path)
            os.makedirs(folder, exist_ok=True)
        if self.save_dtype is not None:
            LOGGER.info("Saving with {} precision.".format(self.save_dtype.__name__))
            values = [a.astype(self.save_dtype) for a in values]
        joblib.dump((variables, finetune_obj), path)

    def load(self, path):
        self.variables, finetune_obj = joblib.load(path)
        finetune_obj.config = get_config(
            error_on_invalid_keywords=False, 
            **dict(finetune_obj.config)
        )
        return finetune_obj

    def initialize_model(self, model: tf.keras.Model):
        transformed_weights: dict[str, np.ndarray] = {}

        if self.variables is not None:
            variables_sv = self.variables
        else:
            variables_sv = dict()

        all_vars = get_fully_qualified_variable_paths(model)
        print({k: v.shape for k, v in sorted(self.fallback.items())})
        print({k: v.shape for k, v in sorted(all_vars.items())})
        for var_name in all_vars.keys():
            saved_var = None
            if var_name in variables_sv.keys():
                saved_var = variables_sv[var_name]
            elif var_name in self.fallback.keys():
                saved_var = self.fallback[var_name]
                            
            if saved_var is not None:
                for func in self.variable_transforms:
                    saved_var = func(var_name, saved_var)
                transformed_weights[var_name] = saved_var
            else:
                if var_name.startswith("model/featurizer"):
                    permitted = self.permit_uninitialized is not None and re.findall(self.permit_uninitialized, var_name)
                    if not permitted:
                        raise ValueError("Uninitialized featurizer variable {}".format(var_name))
        set_weights(model, transformed_weights, all_vars)


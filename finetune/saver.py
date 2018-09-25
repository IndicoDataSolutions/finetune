import re
import os
import warnings
import json
import joblib

import itertools

import numpy as np

from finetune.utils import find_trainable_variables

import tensorflow as tf
SHAPES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'params_shapes.json')
PARAM_PATH = os.path.join(os.path.dirname(__file__), 'model', 'params_{}.npy')
SPECIAL_TOKEN_PATH = os.path.join(os.path.dirname(__file__), "model", "special_tokens.npy")


class Saver:
    def __init__(self, fallback_filename, include_matches=None, exclude_matches=None, variable_transforms=None, save_dtype=None):
        self.variable_transforms = variable_transforms or []
        self.fallback_filename = fallback_filename
        self.include = None if include_matches is None else re.compile(include_matches)
        self.exclude = None if exclude_matches is None else re.compile(exclude_matches)
        self.variables = None
        self.save_dtype = save_dtype
        self.sess = None  # hook out a reference to the session during initialization.

    def save(self, finetune_obj, path, mkdir=True):
        folder = os.path.dirname(path)
        if not os.path.exists(folder) and mkdir:
            os.mkdir(folder)
        if self.fallback_filename is None:
            fallback = dict()
        else:
            fallback = joblib.load(self.fallback_filename)
        included, excluded = self.find_trainable_variables()

        if not all((var.name in fallback) or (var not in tf.trainable_variables()) for var in excluded):
            warnings.warn("Attempting to do a partial save where trainable variables are excluded that do not have a "
                          "corresponding default.")
        values = self.sess.run(included)
        if self.save_dtype is not None:
            values = [a.astype(self.save_dtype) for a in values]

        vars_reduced, vals_reduced = self.remove_unchanged(included, values, fallback)
        var_names = [var.name for var in vars_reduced]
        var_dict = dict(zip(var_names, vals_reduced))
        assert len(vals_reduced) == len(var_names) == len(var_dict)
        joblib.dump((var_dict, finetune_obj), path)

    def load(self, path):
        self.variables, finetune_obj = joblib.load(path)
        return finetune_obj

    def _save_fallback(self):
        with open(SHAPES_PATH) as shapes_file:
            shapes = json.load(shapes_file)
            offsets = np.cumsum([np.prod(shape) for shape in shapes])
            init_params = [np.load(PARAM_PATH.format(n)) for n in range(10)]
            init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
            init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
            init_params[0] = np.load(os.path.join(os.path.dirname(__file__), "model", "embeddings.npy"))
            del init_params[1]
        var_dict = dict(zip((var.name for var in find_trainable_variables("model", exclude="model/target")), init_params))
        joblib.dump(var_dict, self.fallback_filename)

    def get_scaffold_initializer(self):
        """
        Assumes a default init op will be run, this function should be called after all variables are instantiated
        and then the callback run after the graph is finalized
        """
        tf.logging.info("Initializing pre-trained model (PART-1)")
        variables_fb = joblib.load(self.fallback_filename)

        if self.variables is not None:
            variables_sv = self.variables
        else:
            variables_sv = dict()

        all_vars = tf.global_variables()
        init_vals = []
        for var in all_vars:
            var_init = None
            for saved_var_name, saved_var in itertools.chain(variables_sv.items(), variables_fb.items()):
                if saved_var_name == var.name:
                    var_init = (var, saved_var)
                    break

            if var_init is None:
                continue  # Monitored session deals with these.
            else:
                var, saved_var = var_init
                for func in self.variable_transforms:
                    saved_var = func(var.name, saved_var)
                init_vals.append(var.assign(tf.constant(saved_var, dtype=tf.float32)))
        self.variables = None  # not an explicit del but should set reference count to 0 unless being used for deviation regularisation

        def initializer(scafold, sess):
            tf.logging.info("Initializing pre-trained model (PART-2)")
            self.sess = sess
            sess.run(init_vals)
        return initializer

    def get_pretrained_weights(self):
        return joblib.load(self.fallback_filename)

    def remove_unchanged(self, variables, variable_values, fallback_vars):
        skips = []
        for var_val, var in zip(variable_values, variables):
            skip = False
            for fb_var_name, fb_var in fallback_vars.items():
                if fb_var_name == var.name:
                    for func in self.variable_transforms:
                        fb_var = func(var.name, fb_var)
                    if np.allclose(fb_var, var_val):
                        skip = True
                        break
            skips.append(skip)
        return (
            [var for skip, var in zip(skips, variables) if not skip],
            [var_val for skip, var_val in zip(skips, variable_values) if not skip]
        )

    def find_trainable_variables(self):
        trainable_variables = tf.global_variables()
        included = [var for var in trainable_variables if (self.include is None or self.include.match(var.name)) and (
                self.exclude is None or not self.exclude.match(var.name))]
        excluded = [var for var in trainable_variables if var not in set(included)]
        return included, excluded

import os
import joblib

from concurrent.futures import ThreadPoolExecutor

import itertools

import numpy as np
import tensorflow as tf


class Saver:
    def __init__(self, fallback_filename, exclude_matches=None, variable_transforms=None, save_dtype=None):
        self.variable_transforms = variable_transforms or []
        self.fallback_filename = fallback_filename
        self.exclude_matches = exclude_matches
        self.tpe = ThreadPoolExecutor()
        self.fallback_future = self.tpe.submit(joblib.load, fallback_filename)
        self.variables = None
        self.save_dtype = save_dtype
        self.fallback_ = None

    @property
    def fallback(self):
        if self.fallback_ is None:
            self.fallback_ = self.fallback_future.result()
            self.fallback_future = None
            self.tpe.shutdown()
        return self.fallback_

    def save(self, finetune_obj, path, mkdir=True):
        ckpt_reader = tf.train.load_checkpoint(finetune_obj.estimator_dir)
        variable_map = ckpt_reader.get_variable_to_shape_map()
        names = [name for name in variable_map.keys() if self.exclude_matches is None or self.exclude_matches not in name]
        names = [name if name.endswith(":0") else name for name in names]  # strip the :0 off the end
        values = [ckpt_reader.get_tensor(name) for name in names]
        names = [name + ":0" for name in names]

        folder = os.path.dirname(path)
        if not os.path.exists(folder) and mkdir:
            os.mkdir(folder)
        if self.save_dtype is not None:
            values = [a.astype(self.save_dtype) for a in values]

        var_names_reduced, vals_reduced = self.remove_unchanged(names, values, self.fallback)

        var_dict = dict(zip(var_names_reduced, vals_reduced))
        assert len(vals_reduced) == len(var_names_reduced) == len(var_dict)
        joblib.dump((var_dict, finetune_obj), path)

    def load(self, path):
        self.variables, finetune_obj = joblib.load(path)
        return finetune_obj

    def get_scaffold_init_op(self):
        """
        Assumes a default init op will be run, this function should be called after all variables are instantiated
        and then the callback run after the graph is finalized
        """
        if self.variables is not None:
            variables_sv = self.variables
        else:
            variables_sv = dict()

        if tf.contrib.distribute.get_tower_context():
            def assign(var, val):
                def update(var_):
                    return var_.assign(val)

                def merge_fn(dist, vm):
                    return dist.group(dist.update(vm, update))

                tower_context = tf.contrib.distribute.get_tower_context()
                return tower_context.merge_call(merge_fn, var)
        else:
            def assign(var, val):
                return var.assign(val)

        all_vars = tf.global_variables()
        init_vals = []
        default_init = []
        for var in all_vars:
            var_init = None
            for saved_var_name, saved_var in itertools.chain(variables_sv.items(), self.fallback.items()):
                if saved_var_name == var.name:
                    var_init = (var, saved_var)
                    break

            if var_init is None:
                default_init.append(var)
            else:
                var, saved_var = var_init
                for func in self.variable_transforms:
                    saved_var = func(var.name, saved_var)
                init_vals.append(assign(var, saved_var))
        self.variables = None
        init_vals.append(tf.variables_initializer(default_init))
        return tf.group(init_vals)

    def get_pretrained_weights(self):
        return joblib.load(self.fallback_filename)

    def remove_unchanged(self, variable_names, variable_values, fallback_vars):
        skips = []
        for var_val, var_name in zip(variable_values, variable_names):
            skip = False
            for fb_var_name, fb_var in fallback_vars.items():
                if fb_var_name == var_name:
                    for func in self.variable_transforms:
                        fb_var = func(var_name, fb_var)
                    if np.allclose(fb_var, var_val):
                        skip = True
                        break
            skips.append(skip)
        return (
            [var for skip, var in zip(skips, variable_names) if not skip],
            [var_val for skip, var_val in zip(skips, variable_values) if not skip]
        )


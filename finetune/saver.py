import re
import os
import warnings

import joblib

import itertools

import numpy as np

import tensorflow as tf


def partially_matched_in(string, list):
    return any(
        string in val for val in list
    )


def shortest_unique_identifier(var_list, all_vars):
    """
    Seems overkill, will remove all redundant scoping, meaning we can be a little more haphazard with adding scopes.
    """
    variable_names = [var.name for var in var_list]
    other_variables = [var.name for var in all_vars if var not in var_list]
    minimal_var_names = []
    for i, var_name in enumerate(variable_names):
        ref_list = variable_names[:i] + variable_names[i + 1:] + other_variables
        scope_hierachy = var_name.split("/")
        for i in reversed(range(len(scope_hierachy) - 1)):
            attempt = "/".join(scope_hierachy[i:])
            if not partially_matched_in(attempt, ref_list):
                assert attempt in var_name
                minimal_var_names.append(attempt)
                break
    assert len(variable_names) == len(minimal_var_names)
    return minimal_var_names


def remove_unchanged(variables, variable_values, fallback_vars, variable_transforms):
    skips = []
    for var_val, var in zip(variable_values, variables):
        for func in variable_transforms:
            var_val = func(var.name, var_val)
        skip = False
        for fb_var_name, fb_var in fallback_vars.items():
            if fb_var_name in var.name:
                if np.allclose(fb_var, var_val):
                    skip = True
                    break
        skips.append(skip)

    return (
        [var for skip, var in zip(skips, variables) if not skip],
        [var_val for skip, var_val in zip(skips, variable_values) if not skip]
    )


class Saver:
    def __init__(self, fallback_filename, include_matches=None, exclude_matches=None, variable_transforms=None):
        self.variable_transforms = variable_transforms or []
        self.fallback_filename = fallback_filename
        self.include = None if include_matches is None else re.compile(include_matches)
        self.exclude = None if exclude_matches is None else re.compile(exclude_matches)
        self.variables = None

    def save(self, finetune_obj, path):
        if self.fallback_filename is None:
            fallback = dict()
        else:
            fallback = joblib.load(self.fallback_filename)
        included, excluded = self.find_trainable_variables()
        if not all(partially_matched_in(var.name, fallback) for var in excluded):
            warnings.warn("Attempting to do a partial save where variables are excluded that do not have a "
                          "corresponding default.")
        values = finetune_obj.sess.run(included)
        vars_reduced, vals_reduced = remove_unchanged(included, values, fallback)
        shortest_ids = shortest_unique_identifier(vars_reduced, tf.global_variables())
        print(shortest_ids)
        var_dict = dict(zip(shortest_ids, vals_reduced))
        assert len(vals_reduced) == len(shortest_ids) == len(var_dict)
        joblib.dump((var_dict, finetune_obj), path)

    def load(self, path):
        self.variables, finetune_obj = joblib.load(path)
        return finetune_obj

    def initialize(self, sess, expect_new_variables=True):
        """

        :param sess:
        :param expect_new_variables:
        :param variable_transforms: a list of functions with signature, var_name, var_value and return a new var_value. applied in order.
        :return:
        """
        variables_fb = joblib.load(self.fallback_filename)

        if self.variables is not None:
            variables_sv = self.variables
        else:
            variables_sv = dict()

        if expect_new_variables:
            sess.run(tf.global_variables_initializer())
        trainable_variables = tf.global_variables()
        init_vals = []
        for var in trainable_variables:
            var_init = None
            for saved_var_name, saved_var in itertools.chain(variables_fb.items(), variables_sv.items()):
                if saved_var_name in var.name:
                    var_init = (var, saved_var)
                    # Note, there is purposely not a break in here, this loads from saved variables
                    # preferentially to fallback variables
            if var_init is None and expect_new_variables:
                warnings.warn(
                    "Var {} is not found in any checkpoint. Because expect_new_variables is True. The default initializer for this variable is used.".format(
                        var.name))
            elif var_init is None and not expect_new_variables:
                warnings.warn(
                    "Var {} is not found in any checkpoint. Because expect_new_variables is True. This variable will remain uninitialized".format(
                        var.name))
            else:
                var, saved_var = var_init

                for func in self.variable_transforms:
                    saved_var = func(var.name, saved_var)
                init_vals.append(var.assign(saved_var))
        sess.run(init_vals)
        self.variables = None # not an explicit del but should set reference count to 0 unless being used for deviation regularisation

    def get_pretrained_weights(self):
        output = dict()
        variables = joblib.load(self.fallback_filename)
        global_var_names = [var.name for var in tf.global_variables()]
        for gvn in global_var_names:
            for vn in variables:
                if vn in gvn:
                    output[gvn] = variables[vn]
                    break
        return output

    def find_trainable_variables(self):
        trainable_variables = tf.global_variables()
        included = [var for var in trainable_variables if (self.include is None or self.include.match(var.name)) and (
                self.exclude is None or not self.exclude.match(var.name))]
        excluded = [var for var in trainable_variables if var not in set(included)]
        return included, excluded

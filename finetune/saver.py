import re
import warnings

import joblib

import numpy as np

import tensorflow as tf

def partially_matched_in(string, list):
    return any(
        string in val for val in list
    )

def shortest_unique_identifier(var_list):
    """
    Seems overkill, will remove all redundant scoping, meaning we can be a little more haphazard with adding scopes.
    """
    variable_names = [var.name for var in var_list]
    minimal_var_names = []
    for i, var_name in enumerate(variable_names):
        ref_list = variable_names[:i] + variable_names[i+1:]
        scope_hierachy = var_name.split("/")
        for i in reversed(range(len(scope_hierachy) - 1)):
            attempt = "/".join(scope_hierachy[i:])
            if not partially_matched_in(attempt, ref_list):
                assert attempt in var_name
                minimal_var_names.append(attempt)
                break
    assert len(variable_names) == len(minimal_var_names)
    return minimal_var_names


class Saver:
    def __init__(self, output_filename, fallback_filemame, include_matches=None, exclude_matches=None):
        self.output_filename = output_filename
        self.fallback_filename = fallback_filemame
        self.include = None if include_matches is None else re.compile(include_matches)
        self.exclude = None if exclude_matches is None else re.compile(exclude_matches)

    def save(self, sess):
        fallback = joblib.load(self.fallback_filename)
        included, excluded = self.find_trainable_variables()
        if not all(partially_matched_in(var.name, fallback) for var in excluded):
            warnings.warn("Attempting to do a partial save where variables are excluded that do not have a "
                          "corresponding default.")
        values = sess.run(included)
        shortest_ids = shortest_unique_identifier(included)
        var_dict = dict(zip(shortest_ids, values))
        assert len(values) == len(shortest_ids) == len(var_dict)
        joblib.dump(self.output_filename, var_dict)

    def load(self, sess, expect_new_variables=False):
        variables_fb = joblib.load(self.fallback_filename)
        variables_sv = joblib.load(self.output_filename)
        if expect_new_variables:
            sess.run(tf.global_variables_initializer())
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        init_vals = []
        for var in trainable_variables:
            var_init = None
            for saved_var_name, saved_var in variables_fb.items() + variables_sv.items():
                if saved_var_name in var:
                    var_init = (var, saved_var)
                    # Note, there is purposely not a break in here, this loads from saved variables
                    # preferentially to fallback variables
            var, saved_var = var_init
            init_vals = var.assign(saved_var)
        sess.run(init_vals)

    def find_trainable_variables(self):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        included = filter(lambda x: self.include.match(x.name) and not self.exclude.match(x.name), trainable_variables)
        excluded = [var for var in trainable_variables if var not in set(included)]
        return included, excluded




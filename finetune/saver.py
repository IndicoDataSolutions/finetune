import os
from concurrent.futures import ThreadPoolExecutor
import itertools
import logging
import sys

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.python.training import distribution_strategy_context
from tensorflow.contrib.estimator.python.estimator.early_stopping import _StopOnPredicateHook, _get_or_create_stop_var

from finetune.util.estimator import PatchedParameterServerStrategy
from finetune.errors import FinetuneError
from finetune.config import get_config

LOGGER = logging.getLogger('finetune')


class SaverHook(_StopOnPredicateHook):

    def __init__(self, saver, estimator, keep_best_model, early_stopping_steps, steps_per_epoch, eval_frequency):
        super().__init__(self.stop_if_no_metric_improvement_fn, run_every_secs=None,
                         run_every_steps=eval_frequency)
        self.get_current_weights = False
        self.included = None
        self.saver = saver
        self.keep_best_model = keep_best_model
        self.early_stopping_steps = early_stopping_steps or sys.maxsize
        self.steps_per_epoch = steps_per_epoch
        self.estimator = estimator

    def stop_if_no_metric_improvement_fn(self):
        if not self.keep_best_model:
            return False
        eval_results = tf.contrib.estimator.read_eval_metrics(self.estimator.eval_dir())
        if len(eval_results) == 0:
            return False
        most_recent_eval = max(eval_results.items(), key=lambda x: x[0])  # last steps.
        best_eval = min(eval_results.items(), key=lambda x: x[1]["loss"])  # lowest_loss
        if most_recent_eval == best_eval:
            self.get_current_weights = True
        steps_diff = most_recent_eval[0] - best_eval[0]
        tf.logging.info("No improvement in {} steps".format(steps_diff))
        if steps_diff > self.early_stopping_steps and most_recent_eval[0] > self.steps_per_epoch:
            LOGGER.info("No decrease in loss in {} steps, early stopping triggered.".format(steps_diff))
            return True
        return False

    def begin(self):
        super().begin()
        self.included = tf.global_variables()

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        if self.get_current_weights:
            self.saver.variables = dict(zip((var.name for var in self.included), run_context.session.run(self.included)))
            self.get_current_weights = False

    def end(self, session):
        self.stop_if_no_metric_improvement_fn()
        if not self.keep_best_model or self.saver.variables is None or self.get_current_weights:
            self.saver.variables = dict(zip((var.name for var in self.included), session.run(self.included)))


class InitializeHook(tf.train.SessionRunHook):
    def __init__(self, saver):
        self.saver = saver

    def after_create_session(self, session, coord):
        init_fn = self.saver.get_scaffold_init_fn()
        init_fn(None, session)


class Saver:
    def __init__(self, fallback_filename=None, exclude_matches=None, variable_transforms=None, save_dtype=None):
        self.variable_transforms = variable_transforms or []
        self.exclude_matches = exclude_matches
        self.variables = None
        self.save_dtype = save_dtype
        if fallback_filename is not None:
            self.set_fallback(fallback_filename)

    def set_fallback(self, fallback_filename):
        self.tpe = ThreadPoolExecutor()
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

    def get_saver_hook(self, estimator, keep_best_model, steps_per_epoch, early_stopping_steps, eval_frequency):
        return SaverHook(self, estimator=estimator, keep_best_model=keep_best_model, steps_per_epoch=steps_per_epoch,
                         early_stopping_steps=early_stopping_steps, eval_frequency=eval_frequency)

    def save(self, finetune_obj, path, mkdir=True):
        if self.variables is None:
            raise FinetuneError("Cowardly refusing to save default model.")
        if self.exclude_matches is not None:
            variables = {
                k: v for k, v in self.variables.items() if self.exclude_matches not in k
            }
        else:
            variables = self.variables

        names, values = variables.keys(), variables.values()
        folder = os.path.dirname(path)
        if not os.path.exists(folder) and mkdir:
            os.mkdir(folder)
        if self.save_dtype is not None:
            LOGGER.info("Saving with {} precision.".format(self.save_dtype.__name__))
            values = [a.astype(self.save_dtype) for a in values]

        var_names_reduced, vals_reduced = self.remove_unchanged(names, values, self.fallback)
        var_dict = dict(zip(var_names_reduced, vals_reduced))
        assert len(vals_reduced) == len(var_names_reduced) == len(var_dict)
        joblib.dump((var_dict, finetune_obj), path)

    def load(self, path):
        self.variables, finetune_obj = joblib.load(path)
        finetune_obj.config = get_config(**dict(finetune_obj.config))
        return finetune_obj

    def get_scaffold_init_fn(self):
        
        def init_fn(scaffold, session):
            if self.variables is not None:
                variables_sv = self.variables
            else:
                variables_sv = dict()
            all_vars = tf.global_variables()
            self.var_val = []
            for var in all_vars:
                for saved_var_name, saved_var in itertools.chain(variables_sv.items(), self.fallback.items()):
                    if saved_var_name == var.name:
                        for func in self.variable_transforms:
                            saved_var = func(var.name, saved_var)
                        var.load(saved_var, session)
                        break
                            
        return init_fn

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

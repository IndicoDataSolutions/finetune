import os
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import warnings
import re

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.train import SessionRunHook
from tensorflow_estimator.python.estimator.early_stopping import _StopOnPredicateHook

from finetune.errors import FinetuneError
from finetune.config import get_config
from finetune.util.metrics import read_eval_metrics

LOGGER = logging.getLogger("finetune")


def should_be_randomly_initialized(name):
    return "OptimizeLoss" in name or "global_step" in name

class SaverHook(_StopOnPredicateHook):
    def __init__(
        self,
        saver,
        estimator,
        keep_best_model,
        early_stopping_steps,
        steps_per_epoch,
        eval_frequency,
        cache_weights_to_file=False
    ):
        super().__init__(
            self.stop_if_no_metric_improvement_fn,
            run_every_secs=None,
            run_every_steps=eval_frequency,
        )
        self.get_current_weights = False
        self.included = None
        self.saver = saver
        self.keep_best_model = keep_best_model
        self.early_stopping_steps = early_stopping_steps or sys.maxsize
        self.steps_per_epoch = steps_per_epoch
        self.estimator = estimator
        self.cache_weights_to_file = cache_weights_to_file

    def stop_if_no_metric_improvement_fn(self):
        if not self.keep_best_model:
            return False
        eval_results = read_eval_metrics(self.estimator.eval_dir())
        if len(eval_results) == 0:
            return False
        most_recent_eval = max(eval_results.items(), key=lambda x: x[0])  # last steps.
        best_eval = min(eval_results.items(), key=lambda x: x[1]["loss"])  # lowest_loss
        if most_recent_eval == best_eval:
            self.get_current_weights = True
        steps_diff = most_recent_eval[0] - best_eval[0]
        tf.compat.v1.logging.info("No improvement in {} steps".format(steps_diff))

        if (
            steps_diff > self.early_stopping_steps
            and most_recent_eval[0] > self.steps_per_epoch
        ):
            LOGGER.info(
                "Early stopping triggered.".format(
                    steps_diff
                )
            )
            return True
        return False

    def begin(self):
        super().begin()
        self.included = tf.compat.v1.global_variables()
        if self.saver.exclude_matches and not self.cache_weights_to_file:
            self.included = [w for w in self.included if self.saver.exclude_matches not in w.name]

    def _get_weights(self, session):
        if not self.keep_best_model or self.saver.variables is None or self.get_current_weights:
            self.saver.variables = dict(
                zip(
                    (var.name for var in self.included),
                    session.run(self.included),
                )
            )
            if self.cache_weights_to_file:
                joblib.dump(self.saver.variables, os.path.join(self.estimator.eval_dir(), "..", "weights.jl"))
            self.get_current_weights = False

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        if self.get_current_weights:
            self._get_weights(session=run_context.session)

    def end(self, session):
        self.stop_if_no_metric_improvement_fn()
        if not self.keep_best_model or self.saver.variables is None or self.get_current_weights:
            self._get_weights(session=session)


class InitializeHook(SessionRunHook):
    def __init__(self, saver):
        self.saver = saver
        self.init_fn = self.saver.get_scaffold_init_fn()

    def after_create_session(self, session, coord):
        self.init_fn(None, session)

class BatchedVarLoad:
    """
    Basic idea pulled from variable.load. Possible this will change in 2.X so worth tracking what
    that fn changes too and mirror it here.
    """
    def __init__(self):
        self.ops = []
        self.feed = dict()

    def run(self, session):
        session.run(self.ops, feed_dict=self.feed)
        self.ops = []
        self.feed = dict()

    def add(self, var, val):
        if var.name.endswith("we:0") or "bert/embeddings/position_embedding" in var.name: # for backwards comaptibility with pre-saved models
            val = val[:var.shape[0]]
        if hasattr(var, "_values"):
            underlying_vars = var._values
        else:
            underlying_vars = [var]
        for v in underlying_vars:
            if v.initializer not in self.ops:
                self.ops.append(v.initializer)
            self.feed[v.initializer.inputs[1]] = val


class Saver:
    def __init__(
        self,
        fallback_filename=None,
        exclude_matches=None,
        variable_transforms=None,
        save_dtype=None,
        restart_global_step=True,
        permit_uninitialized=None,
        add_tokens=None,
    ):
        self.variable_transforms = variable_transforms or []
        self.exclude_matches = exclude_matches
        self.variables = None
        self.save_dtype = save_dtype
        if fallback_filename is not None:
            self.set_fallback(fallback_filename)
        self.restart_global_step = restart_global_step
        self.permit_uninitialized = permit_uninitialized
        self.add_tokens = add_tokens

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

    def get_saver_hook(
        self,
        estimator,
        keep_best_model,
        steps_per_epoch,
        early_stopping_steps,
        eval_frequency,
        cache_weights_to_file
    ):
        return SaverHook(
            self,
            estimator=estimator,
            keep_best_model=keep_best_model,
            steps_per_epoch=steps_per_epoch,
            early_stopping_steps=early_stopping_steps,
            eval_frequency=eval_frequency,
            cache_weights_to_file=cache_weights_to_file
        )

    def get_initial_step(self):
        if not self.restart_global_step:
            return self.fallback.get("global_step:0", 0)
        return 0

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
        if isinstance(path, str):
            folder = os.path.dirname(path)
            os.makedirs(folder, exist_ok=True)
        if self.save_dtype is not None:
            LOGGER.info("Saving with {} precision.".format(self.save_dtype.__name__))
            values = [a.astype(self.save_dtype) for a in values]

        var_names_reduced, vals_reduced = self.remove_unchanged(
            names, values, self.fallback
        )
        var_dict = dict(zip(var_names_reduced, vals_reduced))
        assert len(vals_reduced) == len(var_names_reduced) == len(var_dict)
        joblib.dump((var_dict, finetune_obj), path)

    def load(self, path):
        self.variables, finetune_obj = joblib.load(path)
        finetune_obj.config = get_config(
            error_on_invalid_keywords=False, 
            **dict(finetune_obj.config)
        )
        return finetune_obj

    def get_scaffold_init_fn(self):

        def init_fn(scaffold, session):
            var_loader = BatchedVarLoad()
            self.var_val = []

            if self.variables is not None:
                variables_sv = self.variables
            else:
                variables_sv = dict()
            all_vars = tf.compat.v1.global_variables()

            global_step_var = tf.compat.v1.train.get_global_step()

            for var in all_vars:
                if self.restart_global_step and global_step_var is not None and global_step_var.name == var.name:
                    continue
                name = var.name
                saved_var = None
                if name in variables_sv.keys():
                    saved_var = variables_sv[name]
                elif name in self.fallback.keys():
                    saved_var = self.fallback[name]
                if saved_var is not None:
                    if self.add_tokens and name == "model/featurizer/shared/shared/weight:0":
                        if var.shape[0] != saved_var.shape[0]:
                            num_rows = var.shape[0] - saved_var.shape[0]
                            new_rows = np.random.normal(size=(num_rows, saved_var.shape[1]), scale=0.01)
                            saved_var = np.concatenate((saved_var, new_rows), axis=0)
                    for func in self.variable_transforms:
                        saved_var = func(name, saved_var)
                    var_loader.add(var, saved_var)
                else:
                    if name.startswith("model/featurizer"):
                        permitted = self.permit_uninitialized is not None and re.findall(self.permit_uninitialized, name)
                        if not permitted:
                            raise ValueError("Uninitialized featurizer variable {}".format(name))
                    
            var_loader.run(session)
        return init_fn

    def remove_unchanged(self, variable_names, variable_values, fallback_vars):
        skips = []
        for var_val, var_name in zip(variable_values, variable_names):
            skip = False
            for fb_var_name, fb_var in fallback_vars.items():
                if fb_var_name == var_name:
                    for func in self.variable_transforms:
                        fb_var = func(var_name, fb_var)
                    if fb_var.shape == var_val.shape and np.allclose(fb_var, var_val):
                        skip = True
                        break
            skips.append(skip)
        return (
            [var for skip, var in zip(skips, variable_names) if not skip],
            [var_val for skip, var_val in zip(skips, variable_values) if not skip],
        )


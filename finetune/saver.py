import os
import collections
from concurrent.futures import ThreadPoolExecutor
import itertools
import logging
import sys

import joblib
import numpy as np
import tensorflow as tf

from tensorflow.python.training import distribution_strategy_context
from tensorflow.contrib.estimator.python.estimator.early_stopping import _StopOnPredicateHook, _get_or_create_stop_var
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary_iterator

from finetune.util.estimator import PatchedParameterServerStrategy
from finetune.errors import FinetuneError

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
        if hasattr(var, "_values"):
            underlying_vars = var._values
        else:
            underlying_vars = [var]
        for v in underlying_vars:
            if v.initializer not in self.ops:
                self.ops.append(v.initializer)
            self.feed[v.initializer.inputs[1]] = val
                                                                                                                                                        
LOGGER = logging.getLogger('finetune')

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'

def _summaries(eval_dir):
    if gfile.Exists(eval_dir):
        for event_file in gfile.Glob(
                os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
            for event in summary_iterator.summary_iterator(event_file):
                yield event
                                      

def read_eval_metrics(eval_dir):
    eval_metrics_dict = collections.defaultdict(dict)
    for event in _summaries(eval_dir):
        if not event.HasField('summary'):
            continue
        metrics = {}
        for value in event.summary.value:
            if value.HasField('simple_value'):
                metrics[value.tag] = value.simple_value
        if metrics:
            eval_metrics_dict[event.step].update(metrics)
    return collections.OrderedDict(
        sorted(eval_metrics_dict.items(), key=lambda t: t[0]))


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
        eval_results = read_eval_metrics(self.estimator.eval_dir())
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
            joblib.dump(self.saver.variables, os.path.join(self.estimator.eval_dir(), "..", "weights.jl"))
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
    def __init__(self, fallback_filename=None, exclude_matches=None, variable_transforms=None, save_dtype=None, target_model_init_from_base_model=False):
        self.variable_transforms = variable_transforms or []
        self.exclude_matches = exclude_matches
        self.variables = None
        self.save_dtype = save_dtype
        if fallback_filename is not None:
            self.set_fallback(fallback_filename)
        self.target_model_init_from_base_model = target_model_init_from_base_model

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
            if self.target_model_init_from_base_model:
                if self.variables is None:
                    self.variables = dict()
                for k, v in self.fallback_.items():
                    self.variables['model/target/' + k] = v
                
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
        return finetune_obj

    def get_scaffold_init_fn(self):
        
        def init_fn(scaffold, session):
            var_loader = BatchedVarLoad()
            self.fallback # force gathering the variables and populating self.variables, this is a hack
            if self.variables is not None:
                variables_sv = self.variables
            else:
                variables_sv = dict()
            all_vars = tf.global_variables()
            init_vals = []
            default_init = []
            self.var_val = []
            print(all_vars)
            for var in all_vars:
                if "global_step" in var.name:
                    continue
                saved_var = variables_sv.get(var.name, self.fallback.get(var.name, None))
                if saved_var is not None and saved_var.shape == tuple(var.get_shape().as_list()):
                    print("init for {}".format(var.name))
                    for func in self.variable_transforms:
                        saved_var = func(var.name, saved_var)
                    print("Success:", var)
                    var_loader.add(var, saved_var)
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
                    if np.allclose(fb_var, var_val):
                        skip = True
                        break
            skips.append(skip)
        return (
            [var for skip, var in zip(skips, variable_names) if not skip],
            [var_val for skip, var_val in zip(skips, variable_values) if not skip]
        )

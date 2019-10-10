import os
from concurrent.futures import ThreadPoolExecutor
import logging
import sys

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.estimator.python.estimator.early_stopping import _StopOnPredicateHook
from tensorflow.estimator import SessionRunHook


from finetune.errors import FinetuneError
from finetune.config import get_config
from finetune.util.metrics import read_eval_metrics


LOGGER = logging.getLogger("finetune")


class SaverHook(_StopOnPredicateHook):
    def __init__(
        self,
        saver,
        estimator,
        keep_best_model,
        early_stopping_steps,
        steps_per_epoch,
        eval_frequency,
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
        self.included = tf.global_variables()

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        if self.get_current_weights:
            self.saver.variables = dict(
                zip(
                    (var.name for var in self.included),
                    run_context.session.run(self.included),
                )
            )
            self.get_current_weights = False

    def end(self, session):
        self.stop_if_no_metric_improvement_fn()
        if (
            not self.keep_best_model
            or self.saver.variables is None
            or self.get_current_weights
        ):
            self.saver.variables = dict(
                zip((var.name for var in self.included), session.run(self.included))
            )


class InitializeHook(SessionRunHook):
    def __init__(self, saver, model_portion="entire_model"):
        self.saver = saver
        self.model_portion = model_portion
        self.need_to_refresh = True  # between predicts of the same model
        self.refresh_base_model = (
            False
        )  # after we have loaded a model that has trained the entire featurizer, we need to reload from fallback for the next model
        self.init_fn = self.saver.get_scaffold_init_fn()

    def after_create_session(self, session, coord):
        if self.model_portion != "entire_model" and self.need_to_refresh:
            if self.model_portion == "target":
                self.init_fn(None, session, self.model_portion)
            else:
                self.init_fn(
                    None, session, "whole_featurizer"
                )  # after_create_session only called at load_featurizer in deployment_model, so load entire featurizer
            self.need_to_refresh = False
        elif self.model_portion == "entire_model":
            self.init_fn(None, session, self.model_portion)

    def before_run(self, run_context):
        if "featurizer" in self.model_portion and (
            self.need_to_refresh or self.refresh_base_model
        ):
            if self.model_portion == "whole_featurizer":
                self.refresh_base_model = True
            self.init_fn(
                None, run_context.session, self.model_portion, self.refresh_base_model
            )
            self.need_to_refresh = False
            self.refresh_base_model = False


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


class Saver:
    def __init__(
        self,
        fallback_filename=None,
        exclude_matches=None,
        variable_transforms=None,
        save_dtype=None,
    ):
        self.variable_transforms = variable_transforms or []
        self.exclude_matches = exclude_matches
        self.variables = None
        self.save_dtype = save_dtype
        if fallback_filename is not None:
            self.set_fallback(fallback_filename)

    def set_fallback(self, fallback_filename):
        self.tpe = ThreadPoolExecutor()
        if not os.path.exists(fallback_filename):
            raise FileNotFoundError("Error loading base model - file not found.")
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
    ):
        return SaverHook(
            self,
            estimator=estimator,
            keep_best_model=keep_best_model,
            steps_per_epoch=steps_per_epoch,
            early_stopping_steps=early_stopping_steps,
            eval_frequency=eval_frequency,
        )

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
        def init_fn(scaffold, session, model_portion=None, refresh_base_model=False):
            var_loader = BatchedVarLoad()
            self.var_val = []
            if self.variables is not None:
                variables_sv = self.variables
            else:
                variables_sv = dict()
            all_vars = tf.global_variables()
            zero_out_adapters = False
            if (
                model_portion != "entire_model"
            ):  # we must be loading in the case of two separate estimators
                all_vars, zero_out_adapters = self.subset_to_load(
                    model_portion, refresh_base_model, all_vars
                )

            for var in all_vars:
                name = var.name
                saved_var = None
                if name in variables_sv.keys():
                    saved_var = variables_sv[name]
                elif name in self.fallback.keys():
                    saved_var = self.fallback[name]

                if zero_out_adapters and "adapter" in name:
                    var_loader.add(var, np.zeros(var.get_shape().as_list()))
                if saved_var is not None:
                    for func in self.variable_transforms:
                        saved_var = func(name, saved_var)
                    var_loader.add(var, saved_var)
            var_loader.run(session)
        return init_fn

    def subset_to_load(self, model_portion, refresh_base_model, all_vars):
        assert model_portion in [
            "featurizer",
            "target",
            "whole_featurizer",
        ], "Must be using separate estimators if loading before graph creation"
        base = [v for v in all_vars if "target" not in v.name]
        zero_out_adapters = False
        if (
            model_portion == "whole_featurizer"
        ):  # load every weight in featurizer - used to initialize and for loading without adapters
            to_load = base
            adapters = [v for v in base if "adapter" in v.name]
            zero_out_adapters = True
        elif (
            model_portion == "featurizer"
        ):  # update featurizer, loading adapters and scaling weights
            norm_variable_scopes = ["b:0", "g:0", "beta:0", "gamma:0"]
            to_load = (
                base
                if refresh_base_model
                else [
                    v
                    for v in base
                    if "target" not in v.name
                    and (
                        "adapter" in v.name
                        or any(scope in v.name for scope in norm_variable_scopes)
                    )
                ]
            )
        elif model_portion == "target":  # update target model weights
            to_load = [v for v in all_vars if "target" in v.name]
        return to_load, zero_out_adapters

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
            [var_val for skip, var_val in zip(skips, variable_values) if not skip],
        )


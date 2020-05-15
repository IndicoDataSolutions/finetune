import os
import traceback
import sys

from tensorflow.core.framework import summary_pb2
from tensorflow.python.summary.writer import writer_cache
import tensorflow as tf
import numpy as np


class InMemoryFinetune(tf.estimator.SessionRunHook):

    def __init__(self, config_to_eval, model, eval_dir, X, Y, X_test, Y_test, name=None, every_n_iter=100):
        if every_n_iter is None or every_n_iter <= 0:
            raise ValueError('invalid every_n_iter=%s.' % every_n_iter)

        self._current_finetune = model
        self._config_to_finetune = config_to_eval
        self._name = name
        self._every_n_iter = every_n_iter
        self._timer = tf.estimator.SecondOrStepTimer(every_steps=every_n_iter)
        self._eval_dir = eval_dir
        self.train_data = (X, Y)
        self.test_data = (X_test, Y_test)
        self._iter_count = 0

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def after_create_session(self, session, coord):
        """Does first run which shows the metrics before training."""
        self._evaluate(session)

    def _evaluate(self, session):
        try:
            with tf.Graph().as_default():
                from finetune import Classifier
                model = Classifier(**self._config_to_finetune)

                if self._current_finetune.saver.variables:
                    model.saver.variables = {
                        k: v.copy() for k, v in self._current_finetune.saver.variables.items()
                        if "global_step" not in k and "Adam" not in k
                    }

                model.saver.fallback_ = {
                    k: v for k, v in self._current_finetune.saver.fallback.items() if "global_step" not in k
                }
                train_x, train_y = self.train_data
                model.fit(train_x, train_y)
                test_x, test_y = self.test_data
                test_accuracy = np.mean(model.predict(test_x) == test_y)
                train_accuracy = np.mean(model.predict(train_x) == train_y)
        except IOError as e:
            traceback.print_exc(file=sys.stdout)
            test_accuracy = -1.0
            train_accuracy = -1.0

        global_step = session.run(tf.compat.v1.train.get_or_create_global_step())
        directory = os.path.join(self._eval_dir, "..", "finetuning")

        if not os.path.exists(directory):
            os.makedirs(directory)
        summary_writer = writer_cache.FileWriterCache.get(directory)
        summary_proto = summary_pb2.Summary()
        summary_proto.value.add(tag="finetuning/{}_train_accurary".format(self._name), simple_value=float(train_accuracy))
        summary_proto.value.add(tag="finetuning/{}_test_accurary".format(self._name), simple_value=float(test_accuracy))
        summary_writer.add_summary(summary_proto, global_step)
        summary_writer.flush()
    
        self._timer.update_last_triggered_step(self._iter_count)

    def after_run(self, run_context, run_values):
        self._iter_count += 1
        if self._timer.should_trigger_for_step(self._iter_count):
            self._evaluate(run_context.session)

    def end(self, session):
        self._evaluate(session)

def make_in_memory_finetune_hooks(model, estimator):
    hooks = []
    for f in model.config.in_memory_finetune:
        hooks.append(InMemoryFinetune(
            config_to_eval=f["config"],
            model=model,
            eval_dir=estimator.eval_dir(),
            X=f["X"],
            Y=f["Y"],
            X_test=f["X_test"],
            Y_test=f["Y_test"],
            name=f["name"],
            every_n_iter=f["every_n_iter"]
        ))
    return hooks

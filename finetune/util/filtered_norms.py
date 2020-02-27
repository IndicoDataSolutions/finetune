import os
import traceback
import sys

from tensorflow.core.framework import summary_pb2
from tensorflow.python.summary.writer import writer_cache
import tensorflow as tf
import numpy as np
import math

from finetune.util.metrics import summary_macro_f1


class FilteredNormsHook(tf.train.SessionRunHook):

    def __init__(self, filter, model, eval_dir):
        self.filter = filter
        print('filter', filter)
        self.model = model
        self._eval_dir = eval_dir
        self._timer = tf.train.SecondOrStepTimer(every_steps=1)
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
                if self.model.saver.variables:
                    model_vars = {k: v for k, v in self.model.saver.variables.items() if "global_step" not in k and "Adam" not in k}
                else:
                    model_vars = {k: v for k, v in self.model.saver.fallback_.items() if "global_step" not in k and "Adam" not in k}
                print(self.filter)
                if self.filter[0] == '!':
                    variables = np.array([v for k, v in model_vars.items() if self.filter not in k])
                else:
                    variables = np.array([v for k, v in model_vars.items() if self.filter in k])

                weight_norm = math.sqrt(np.sum([np.linalg.norm(v) ** 2 for v in variables]))
                
        except IOError as e:
            traceback.print_exc(file=sys.stdout)
            grad_norm = -1.0
            weight_norm = -1.0

        global_step = session.run(tf.train.get_or_create_global_step())
        directory = os.path.join(self._eval_dir)

        if not os.path.exists(directory):
            os.makedirs(directory)
        summary_writer = writer_cache.FileWriterCache.get(directory)
        summary_proto = summary_pb2.Summary()
        summary_proto.value.add(tag="norms/{}_weight_norm".format(self.filter), simple_value=weight_norm)
        summary_writer.add_summary(summary_proto, global_step)
        print('{} weight norm: {}'.format(self.filter, weight_norm))
        summary_writer.flush()
    
        self._timer.update_last_triggered_step(self._iter_count)

    def after_run(self, run_context, run_values):
        self._iter_count += 1
        if self._timer.should_trigger_for_step(self._iter_count):
            self._evaluate(run_context.session)

    def end(self, session):
        self._evaluate(session)

def make_filtered_norm_hooks(model, estimator):
    hooks = []
    for filter in model.config.filtered_norms:
        hooks.append(FilteredNormsHook(
            filter=filter,
            model=model,
            eval_dir=estimator.eval_dir()
        ))
    return hooks
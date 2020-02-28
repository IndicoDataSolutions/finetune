import tensorflow as tf
import re

def make_summary_optimizer(optimizer, regex):
    class SummaryOptimizer(optimizer):
        def apply_gradients(self, grads_and_vars, *args, **kwargs):
            for g, v in grads_and_vars:
                if re.match(regex, v.name):
                    tf.summary.scalar("weight_norm/{}".format(v.name), tf.norm(v))
                    tf.summary.scalar("grad_norm/{}".format(v.name), tf.norm(g))
            return super().apply_gradients(grads_and_vars, *args, **kwargs)
    return SummaryOptimizer

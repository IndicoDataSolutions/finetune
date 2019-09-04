from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.opt.python.training.weight_decay_optimizers import AdamWOptimizer as AdamWOptimizerBroken

import tensorflow as tf


class AdamWOptimizer(AdamWOptimizerBroken):
    def _resource_scatter_add(self, x, i, v, _=None):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return tf.convert_to_tensor(x)

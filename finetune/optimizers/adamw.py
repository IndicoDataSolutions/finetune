from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.opt.python.training.weight_decay_optimizers import AdamWOptimizer as AdamWOptimizerBroken

import tensorflow as tf


class AdamWOptimizer(AdamWOptimizerBroken):
    def _resource_scatter_add(self, x, i, v, _=None):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return tf.convert_to_tensor(x)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None, decay_var_list=None):
        new_grads_and_vars = [None] * len(grads_and_vars)
        for i, pair in enumerate(grads_and_vars):
            if 'model/target' in pair[1].name:
                new_grads_and_vars[i] = (pair[0] * self.target_model_lr_mult, pair[1])
            else:
                new_grads_and_vars[i] = (pair[0], pair[1])
        return super().apply_gradients(new_grads_and_vars, global_step=global_step, name=name, decay_var_list=decay_var_list)

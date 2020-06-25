import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops


class AdamW(tf.compat.v1.train.AdamOptimizer):
    def __init__(self, learning_rate, *, weight_decay, decay_var_list, **kwargs):
        """Extension class that adds weight decay to an optimizer.
        Args:
            weight_decay: A `Tensor` or a floating point value, the factor by
                which a variable is decayed in the update step.
            **kwargs: Optional list or tuple or set of `Variable` objects to
                decay.
        """
        self.wd = weight_decay
        self.decay_var_list = decay_var_list
        super().__init__(learning_rate, **kwargs)

    def _decay_weights_op(self, var):
        if self.decay_var_list is None or var in self.decay_var_list:
            return var.assign_sub(self.wd * var, self._use_locking)
        return tf.no_op()

    def _decay_weights_sparse_op(self, var, indices):
        if self.decay_var_list is None or var in self.decay_var_list:
            update = -self.wd * tf.gather(var, indices)
            return self._resource_scatter_add(var, indices, update)
        return tf.no_op()

    def _resource_apply_dense(self, grad, var):
        with tf.control_dependencies([self._decay_weights_op(var)]):
            return super()._resource_apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        decay_op = self._decay_weights_sparse_op(var, indices)
        with tf.control_dependencies([decay_op]):
            return super()._resource_apply_sparse(grad, var, indices)

    def _resource_scatter_add(self, x, i, v):
        with tf.control_dependencies(
            [resource_variable_ops.resource_scatter_add(x.handle, i, v)]
        ):
            return tf.identity(x)

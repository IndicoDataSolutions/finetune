import tensorflow as tf

def wrap_optimizer(keras_optimizer):
    class WrappedOptimizer(keras_optimizer):
        def compute_gradients(self, loss, variables, colocate_gradients_with_ops=False):
            gradients = tf.gradients(
                loss,
                variables,
                name='gradients',
                colocate_gradients_with_ops=colocate_gradients_with_ops,
            )
            return zip(gradients, variables)
    return wrap_optimizer
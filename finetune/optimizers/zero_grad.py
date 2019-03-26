import types
import tensorflow as tf


def dont_optimize_zeros(optimizer):
    """
    Does not apply the optimiser to variables with 0 gradient. This stops accumulators being diluted in adam and does
     not apply weight decay to these either in the case of weight decay optimisers.

    Based on:
        tf.contrib.opt.MultitaskOptimizerWrapper

    :param optimizer:
    :return:
    """
    
    def _is_all_zeros(grad):
        return tf.equal(tf.count_nonzero(grad), 0)
    
    overridden_methods = ('_apply_dense', '_resource_apply_dense', '_apply_sparse', '_resource_apply_sparse')

    def _get_wrapper(fn, opt):
        
        def wrap(self, grad, *args, **kwargs):
            all_zeros = _is_all_zeros(grad)

            def call_fn():
                with tf.control_dependencies([fn(grad, *args, **kwargs)]):
                    return tf.no_op()
            return tf.cond(all_zeros, tf.no_op, call_fn)
        
        wrapper = types.MethodType(wrap, opt)
        return wrapper

    for name in overridden_methods:
        fn = getattr(optimizer, name)
        wrapper = _get_wrapper(fn, optimizer)
        setattr(optimizer, name, wrapper)

    return optimizer

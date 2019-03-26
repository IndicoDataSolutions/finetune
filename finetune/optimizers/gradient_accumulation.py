import tensorflow as tf


def get_grad_accumulation_optimizer(optimizer_class, accum_steps):
    """
    Adds gradient accumulation to an Optimizer.

    Unlike most optimizers, this will create a global step if one does not already exist.
    This optimizer also increments this global step automatically, whether a step is passed or not.

    :param optimizer_class: A subclass of tf.train.Optimizer to add gradient accumulation too.
    :param accum_steps: An int value determining how many gradients to accumulate before performing an optimizer step.

    :return: A new Optimizer, with gradient accumulation.
    """

    class GradAccumulationOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def apply_gradients(self, grads_and_vars, global_step=None, *args, **kwargs):
            add_gradients_ops = []
            accumulation_vars = []
            grads_and_accumulated_vars = []
            for g, v in grads_and_vars:
                accum_grad = tf.get_variable(
                    name=g.name[:-2] + "_acc",
                    shape=g.shape,
                    dtype=g.dtype,
                    initializer=tf.constant_initializer(0),
                    use_resource=True,
                    trainable=False
                )
                add_gradients_ops.append(accum_grad.assign_add(g))
                accumulation_vars.append(accum_grad)

                grads_and_accumulated_vars.append((accum_grad, v))

            global_step = global_step if global_step is not None else tf.train.get_or_create_global_step()

            with tf.control_dependencies(add_gradients_ops):
                def apply_grads():
                    with tf.control_dependencies([
                        super(GradAccumulationOptimizer, self).apply_gradients(
                            grads_and_accumulated_vars, global_step=global_step, *args, **kwargs
                        )
                    ]):
                        apply_grads_op = tf.group(*[g.assign(g.initial_value) for g in accumulation_vars])
                    return apply_grads_op

                return tf.cond(
                    tf.equal(global_step % accum_steps, accum_steps - 1),
                    true_fn=apply_grads,
                    false_fn=lambda: tf.group(global_step.assign_add(1))
                )

    return GradAccumulationOptimizer

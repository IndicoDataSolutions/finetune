import functools

import tensorflow as tf

from finetune.optimizers.adafactor import AdafactorWOptimizer, AdafactorOptimizer
from finetune.optimizers.adamw import AdamWOptimizer
from finetune.optimizers.gradient_accumulation import get_grad_accumulation_optimizer
from finetune.optimizers.learning_rate_schedules import schedules

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
    "global_gradient_norm",
]

OPTIMIZERS = {
    "AdamW": AdamWOptimizer,
    "AdafactorW": AdafactorWOptimizer,
    "Adafactor": AdafactorOptimizer,
}


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
    return list(zip(clipped_gradients, variables))


def get_optimizer(
    optimizer_name, learning_rate, b1, b2, epsilon, l2_reg, vector_l2, accumulate_steps, scale_loss
):
    Optimizer = OPTIMIZERS.get(optimizer_name, None)
    if Optimizer is None:
        raise FinetuneError(
            "Optimizer must be in {}, not {}".format(
                list(OPTIMIZERS.keys()), optimizer_name
            )
        )

    if accumulate_steps > 1:
        Optimizer = get_grad_accumulation_optimizer(Optimizer, accumulate_steps)

    decay_var_list = [
        v for v in tf.global_variables() if len(v.get_shape()) > 1 or vector_l2
    ]

    opt = Optimizer(
        learning_rate=learning_rate,
        beta1=b1,
        beta2=b2,
        epsilon=epsilon,
        weight_decay=l2_reg * learning_rate,
    )

    opt.apply_gradients = functools.partial(
        opt.apply_gradients, decay_var_list=decay_var_list
    )

    if scale_loss:
        opt = tf.train.experimental.MixedPrecisionLossScaleOptimizer(opt, "dynamic")
    return opt


def optimize_loss(
    loss,
    learning_rate,
    optimizer_name,
    clip_gradients,
    lr_schedule,
    lr_warmup,
    total_num_steps,
    summarize_grads,
    scale_loss,
    b1,
    b2,
    epsilon,
    l2_reg,
    vector_l2,
    accumulate_steps,
    colocate_gradients_with_ops=True,
):
    global_step = tf.train.get_or_create_global_step()

    with tf.compat.v1.variable_scope("OptimizeLoss", [loss, global_step]):
        tf.summary.scalar("loss", loss)

        training_fraction = tf.to_float(global_step) / total_num_steps
        learning_rate = tf.maximum(
            0.0,
            learning_rate * schedules[lr_schedule](training_fraction, warmup=lr_warmup)
        )
        tf.summary.scalar("learning_rate", learning_rate)

        opt = get_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            b1=b1, 
            b2=b2,
            epsilon=epsilon,
            l2_reg=l2_reg,
            vector_l2=vector_l2,
            accumulate_steps=accumulate_steps,
            scale_loss=scale_loss
        )
        variables = tf.compat.v1.trainable_variables()

        # Compute gradients.
        gradients = opt.compute_gradients(
            loss, variables, colocate_gradients_with_ops=colocate_gradients_with_ops
        )
        tf.summary.scalar(
            "global_norm/gradient_norm", tf.global_norm(list(zip(*gradients))[0]),
        )

        gradients = _clip_gradients_by_norm(gradients, clip_gradients)
        tf.summary.scalar(
            "global_norm/clipped_gradient_norm",
            tf.global_norm(list(zip(*gradients))[0]),
        )

        # Add histograms for variables, gradients and gradient norms.
        if summarize_grads:
            for gradient, variable in gradients:
                if isinstance(gradient, ops.IndexedSlices):
                    grad_values = gradient.values
                else:
                    grad_values = gradient

                if grad_values is not None:
                    var_name = variable.name.replace(":", "_")
                    tf.summary.histogram("gradients/%s" % var_name, grad_values)
                    tf.summary.scalar(
                        "gradient_norm/%s" % var_name, tf.global_norm([grad_values]),
                    )


        # Create gradient updates.
        grad_updates = opt.apply_gradients(gradients, global_step=global_step, name="train")

    return grad_updates

import logging
import tensorflow as tf

from finetune.optimizers.adafactor import AdafactorOptimizer

from finetune.optimizers.gradient_accumulation import get_grad_accumulation_optimizer
from finetune.optimizers.learning_rate_schedules import schedules
from finetune.optimizers.weight_decay import AdamW

LOGGER = logging.getLogger("finetune")


OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
    "global_gradient_norm",
]

OPTIMIZERS = {"AdamW": AdamW, "Adafactor": AdafactorOptimizer}


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
    return list(zip(clipped_gradients, variables))


def get_optimizer(
    optimizer_name,
    learning_rate,
    b1,
    b2,
    epsilon,
    l2_reg,
    vector_l2,
    accumulate_steps,
    mixed_precision,
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
        v
        for v in tf.compat.v1.global_variables()
        if (len(v.get_shape()) > 1 or vector_l2) and "Transition_matrix" not in v.name
    ]

    opt = Optimizer(
        learning_rate=learning_rate,
        beta1=b1,
        beta2=b2,
        epsilon=epsilon,
        weight_decay=l2_reg * learning_rate,
        decay_var_list=decay_var_list,
    )

    if mixed_precision:
        opt = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(
            # Default is 2k which is too high for some of our small datasets. May be sitting at underflowing gradients for large amounts of training.
            opt,
            tf.compat.v1.train.experimental.DynamicLossScale(
                increment_period=100
            ),
        )
    return opt

def is_numerical_tensor(t):
    return t.dtype != tf.string


def optimize_loss(
    loss,
    learning_rate,
    optimizer_name,
    clip_gradients,
    lr_schedule,
    lr_warmup,
    total_num_steps,
    summarize_grads,
    mixed_precision,
    b1,
    b2,
    epsilon,
    l2_reg,
    vector_l2,
    accumulate_steps,
    max_training_hours=None,
    colocate_gradients_with_ops=True,
):
    global_step = tf.compat.v1.train.get_or_create_global_step()

    with tf.compat.v1.variable_scope("OptimizeLoss", [loss, global_step]):
        tf.compat.v1.summary.scalar("loss", loss)

        training_fraction = tf.cast(global_step, dtype=tf.float32) / total_num_steps
        if max_training_hours is not None:
            start_time = tf.compat.v1.get_variable(
                initializer=tf.timestamp,
                trainable=False,
                name="start_time",
                dtype=tf.float32,
            )
            training_fraction = tf.minimum(
                tf.maximum(
                    training_fraction,
                    tf.cast((tf.timestamp() - start_time) / (max_training_hours * 3600), tf.float32),
                ),
                1.0,
            )
        training_fraction_var = tf.compat.v1.get_variable(
            initializer=-1.0,
            trainable=False,
            name="training_fraction",
            dtype=tf.float32,
        )
        learning_rate = tf.maximum(
            0.0,
            learning_rate * schedules[lr_schedule](training_fraction, warmup=lr_warmup),
        )
        tf.compat.v1.summary.scalar("learning_rate", learning_rate)

        opt = get_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            epsilon=epsilon,
            l2_reg=l2_reg,
            vector_l2=vector_l2,
            accumulate_steps=accumulate_steps,
            mixed_precision=mixed_precision,
        )
        variables = tf.compat.v1.trainable_variables()
        variables = [v for v in variables if is_numerical_tensor(v)]

        # Compute gradients.
        gradients = list(
            zip(tf.gradients(ys=loss, xs=variables, name="gradients"), variables)
        )
        for g, v in gradients:
            if g is None:
                LOGGER.warning("Variable {} has None grads".format(v.name))

        tf.compat.v1.summary.scalar(
            "global_norm/gradient_norm",
            tf.linalg.global_norm([g[0] for g in gradients]),
        )

        gradients = _clip_gradients_by_norm(gradients, clip_gradients)
        tf.compat.v1.summary.scalar(
            "global_norm/clipped_gradient_norm",
            tf.linalg.global_norm(list(zip(*gradients))[0]),
        )

        # Add histograms for variables, gradients and gradient norms.
        if summarize_grads:
            for gradient, variable in gradients:
                if isinstance(gradient, tf.IndexedSlices):
                    grad_values = gradient.values
                else:
                    grad_values = gradient

                if grad_values is not None:
                    var_name = variable.name.replace(":", "_")
                    tf.compat.v1.summary.histogram(
                        "gradients/%s" % var_name, grad_values
                    )
                    tf.compat.v1.summary.scalar(
                        "gradient_norm/%s" % var_name,
                        tf.linalg.global_norm([grad_values]),
                    )

        # Create gradient updates.
        with tf.compat.v1.control_dependencies(
            [global_step.assign_add(1), training_fraction_var.assign(training_fraction)]
        ):
            grad_updates = opt.apply_gradients(gradients, name="train")

    return grad_updates

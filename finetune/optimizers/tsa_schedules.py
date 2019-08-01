import tensorflow as tf
from finetune.errors import FinetuneError

# From Unsupervised Data Augmentation for Consistency Training, Xie et al. 2019
def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    if schedule not in ["linear_schedule", "exp_schedule", "log_schedule"]:
        raise FinetuneError(
            'Invalid TSA schedule set in config. Must be one of ["linear_schedule", "exp_schedule", "log_schedule"].'
        )
    training_progress = tf.to_float(global_step) / tf.to_float(num_train_steps)
    if schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = tf.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - tf.exp((-training_progress) * scale)
    return threshold * (end - start) + start


def tsa_loss(n_targets, config, clf_losses, clf_logits, targets):
    with tf.variable_scope("tsa"):
        start = 1 / n_targets
        global_step = tf.train.get_or_create_global_step()
        total_num_steps = config.n_epochs * config.dataset_size // config.batch_size
        tsa_threshold = get_tsa_threshold(
            config.tsa_schedule, global_step, total_num_steps, start, 1
        )

        clf_logits = tf.nn.log_softmax(clf_logits)
        multiplied = targets * tf.exp(clf_logits)
        correct_label_probs = tf.reduce_sum(multiplied, axis=1)
        larger_than_threshold = tf.greater(correct_label_probs, tsa_threshold)
        loss_mask = tf.ones_like(clf_losses, dtype=clf_losses.dtype)

        loss_mask = loss_mask * (1 - tf.cast(larger_than_threshold, tf.float32))
        loss_mask = tf.stop_gradient(loss_mask)
        clf_losses = tf.multiply(clf_losses, loss_mask)

        return clf_logits, clf_losses

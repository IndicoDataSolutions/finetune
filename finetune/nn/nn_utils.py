import tensorflow as tf
from finetune.util.shapes import shape_list
from tensorflow_addons.text.crf import crf_log_likelihood


def norm(x, scope, axis=[-1], e=1e-5):
	with tf.compat.v1.variable_scope(scope):
		n_state = shape_list(x)[-1]
		g = tf.compat.v1.get_variable('g', [n_state], initializer=tf.compat.v1.constant_initializer(1))
		b = tf.compat.v1.get_variable('b', [n_state], initializer=tf.compat.v1.constant_initializer(0))
		u = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=True)
		s = tf.reduce_mean(input_tensor=tf.square(x - u), axis=axis, keepdims=True)
		x = (x - u) * tf.math.rsqrt(s + e)
		x = x * g + b
		return x


def dropout(x, pdrop, train):
	if train and pdrop > 0:
		x = tf.nn.dropout(x, 1 - (1 - pdrop))
		return x


def build_ema_getter(name_scope_name, decay=0.999):
    """
    Builds exponential moving average copies of all trainable variables, then
    creates a custom getter that retrieves these variables
    """
    with tf.compat.v1.name_scope(name_scope_name + "/ema_variables"):
        original_trainable_vars = {
            tensor.op.name: tensor
            for tensor
            in
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        }
        ema = tf.train.ExponentialMovingAverage(decay)
        update_op = ema.apply(original_trainable_vars.values())
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update_op)

    def use_ema_variables(getter, name, *args, **kwargs):
        assert name in original_trainable_vars, "Unknown variable {}.".format(name)
        ret = ema.average(original_trainable_vars[name])
        return ret

    return use_ema_variables

def tsa_log_schedule(x, minimium=0.25):
    """
    Log training signal annealing
    """
    a = 1 - tf.exp(-x * 5)
    return a * (1 - minimium) + minimium

def tsa_filter(method, loss, logits, use_crf, total_steps):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    training_fraction = tf.cast(global_step, dtype=tf.float32) / total_steps
    tsa_thresh = tsa_log_schedule(training_fraction, minimium=0.000001)
    tsa_thresh = 0
    if use_crf:
        # Loss for CRF is -log_likelihood, so we can use it to get the
        # probability of the sequences
        seq_probs = tf.exp(-loss)
    else:
        token_probs = tf.reduce_max(tf.nn.softmax(logits, axis=-1),
                                    axis=-1)
        seq_probs = tf.reduce_mean(token_probs, axis=-1)
    # Keep only sequences with prob under threshhold
    mask = tf.less(seq_probs, tsa_thresh)
    mask = tf.compat.v1.Print(mask, [mask, tsa_thresh, seq_probs, loss])
    return loss * tf.stop_gradient(tf.cast(mask, tf.float32)), tsa_thresh

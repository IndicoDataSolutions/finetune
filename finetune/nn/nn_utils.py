import tensorflow as tf
from finetune.util.shapes import shape_list


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
        # ret = tf.compat.v1.Print(ret, [name, ret, ret.op.name], summarize=3)
        return ret

    return use_ema_variables

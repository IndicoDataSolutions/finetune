import math
import tensorflow as tf
from itertools import zip_longest


def warmup_cosine(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1


def warmup_linear(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)


schedules = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


def AdamWeightDecay(params, grads, lr, schedule, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False, max_grad_norm=-1, pretrained_weights=None, deviation_regularization=0, **kwargs):
    """
    Adam with weight decay fix and added weight decay to pre-trained weights.
    """
    
    with tf.variable_scope('adam', reuse=tf.AUTO_REUSE):
        t = tf.get_variable("t", shape=1, initializer=tf.zeros_initializer() , dtype=tf.float32, trainable=False)
        tt = t + 1
        updates = [t.assign(tt)]
        if max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

        for p, g, ptw, msk in zip_longest(params, grads, pretrained_weights["init_params"], pretrained_weights["mask"]):

            if g is None:
                print("can't train", p.name, g)
            else:
                if isinstance(g, tf.IndexedSlices):
                    g = tf.convert_to_tensor(g)
                prefix = p.name.split(':')[0]
                m = tf.get_variable(prefix + "_m", shape=p.get_shape(), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
                v = tf.get_variable(prefix + "_v", shape=p.get_shape(), dtype=tf.float32,
                                    initializer=tf.zeros_initializer(), trainable=False)

                lrt = lr * tf.sqrt(1 - b2 ** tt) / (1 - b1 ** tt)
                lrt *= schedule(t / t_total)
                mt = b1 * m + (1 - b1) * g
                vt = b2 * v + (1 - b2) * g * g

                update_vec = mt / (tf.sqrt(vt) + e)

                if (len(p.get_shape()) > 1 or vector_l2) and l2 > 0:
                    update_vec += l2 * p

                if ptw is not None and deviation_regularization > 0:
                    update_vec += deviation_regularization * (msk * (p - ptw) + (1 - msk) * p)

                pt = p - lrt * update_vec

                updates.extend([m.assign(mt), v.assign(vt), p.assign(pt)])
        return tf.group(*updates)

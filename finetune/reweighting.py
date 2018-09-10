import tensorflow as tf


def learning_to_reweight(gold_data, gold_targets, data, targets, model, loss, lr=1e-5):
    """
    An implementation of https://arxiv.org/pdf/1803.09050.pdf

        "Learning to reweight examples for robust deep learning" - M Ren et al.
    """

    def net(data):
        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            return model(data)

    def meta_net(data):
        with tf.variable_scope("meta_net", reuse=tf.AUTO_REUSE):
            return model(data)

    # Lines 4 - 5 initial forward pass to compute the initial weighted loss
    y_f_hat = net(data)
    y_f_hat_meta = meta_net(data)

    net_vars = tf.global_variables(scope="net")
    meta_net_vars = tf.global_variables(scope="meta_net")
    re_init_vars = []
    for n_v, met_v in zip(net_vars, meta_net_vars):
        re_init_vars.append(met_v.assign(n_v))

    with tf.control_dependencies(re_init_vars):
        cost = loss(y_f_hat_meta, targets)
    eps = tf.zeros_like(cost)
    l_f_meta = tf.reduce_sum(cost * eps)

    # Line 6 perform a parameter update
    grads = tf.gradients(l_f_meta, meta_net_vars)
    patch_dict = dict()
    for grad, var in zip(grads, meta_net_vars):
        if grad is None:
            print("None grad for variable {}".format(var.name))
        else:
            print("Gradient found for variable {}".format(var.name))
            patch_dict[var.name] = -grad * lr

    # Monkey patch get_variable
    old_get_variable = tf.get_variable

    def _get_variable(*args, **kwargs):
        var = old_get_variable(*args, **kwargs)
        print(var.name)
        print(patch_dict)
        return var + patch_dict.get(var.name, 0.0)

    tf.get_variable = _get_variable

    # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
    y_g_hat = meta_net(gold_data)

    tf.get_variable = old_get_variable

    l_g_meta = loss(y_g_hat, gold_targets)

    grad_eps_es = tf.gradients(l_g_meta, eps)[0]

    # Line 11 computing and normalizing the weights
    w_tilde = tf.maximum(-grad_eps_es, 0.)
    norm_c = tf.reduce_sum(w_tilde)

    w = w_tilde / (norm_c + tf.cast(tf.equal(norm_c, 0.0), dtype=tf.float32))

    # Lines 12 - 14 computing for the loss with the computed weights
    # and then perform a parameter update
    cost = loss(y_f_hat, targets)
    l_f = tf.reduce_sum(cost * w)

    logits = y_f_hat
    loss = l_f

    return logits, loss
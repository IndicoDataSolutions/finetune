import tensorflow as tf	
from finetune.nn.nn_utils import dropout, norm
from finetune.util.shapes import shape_list


def embed_context(context, featurizer_state, config, train):
    with tf.variable_scope("context_embedding"):
        context_dim = shape_list(context)[-1]
        context_weight = tf.get_variable(
            name="ce",	
            shape=[context_dim, config.n_context_embed],
            initializer=tf.random_normal_initializer(stddev=config.context_embed_stddev),	
        )
        context_bias = tf.get_variable(
            name="ca",	
            shape=[config.n_context_embed],	
            initializer=tf.zeros_initializer(),	
        )
        c_embed = tf.add(tf.tensordot(context, context_weight, axes=[[-1], [0]]), context_bias)
    featurizer_state['context'] = c_embed
    return featurizer_state
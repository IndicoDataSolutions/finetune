import tensorflow as tf	
from finetune.nn.nn_utils import dropout, norm
from finetune.util.shapes import shape_list


def embed_context(context, featurizer_state, config, train):
    context_dim = shape_list(context)[-1]
    context_weight = tf.get_variable(
        name="ce",	
        shape=[context_dim, config.n_context_embed],	
        initializer=tf.random_normal_initializer(stddev=config.weight_stddev),	
    )
    context_bias = tf.get_variable(
        name="ca",	
        shape=[config.n_context_embed],	
        initializer=tf.zeros_initializer(),	
    )
    with tf.variable_scope("context_embedding"):
        c_embed = tf.add(tf.multiply(context, context_weight), context_bias)
    featurizer_state['context'] = c_embed
    return featurizer_state
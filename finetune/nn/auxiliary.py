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


def positional_embed_context(context, featurizer_state, config, train):
    with tf.variable_scope('positional_embed'):
        
def pairwise_embed_context(context, featurizer_state, config, train):
    with tf.variable_scope("context_attn_embedding"):
        context_dim = shape_list(context)[-1]
        diff = tf.expand_dims(context, 1) - tf.expand_dims(context, 2)
        g = tf.get_variable(
            name='g',
            shape=[1, config.n_heads, 1, 1, context_dim],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=config.context_embed_stddev)
        )
        b = tf.get_variable(
            name='b',
            shape=[1, config.n_heads, 1, 1, context_dim],
            initializer=tf.zeros_initializer()
        )
        proximity = tf.nn.sigmoid(-diff)
        proximity = tf.expand_dims(proximity, axis=1)
        offset = proximity * g + b
        total_offset = tf.reduce_sum(offset, axis=-1, keep_dims=False) 
    featurizer_state['context'] = total_offset
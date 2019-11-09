import tensorflow as tf	
from finetune.nn.nn_utils import dropout, norm
from finetune.util.shapes import shape_list


def embed_context(context, featurizer_state, config, train):
    # context.set_shape([None, config.max_length, context_dim])
    context_dim = shape_list(context)[-1]
    context_embed_weights = tf.get_variable(	
        name="ce",	
        shape=[context_dim, config.n_context_embed],	
        initializer=tf.random_normal_initializer(stddev=config.weight_stddev),	
    )	

    context_weighted_avg = tf.get_variable(	
        name="cwa",
        shape=[context_dim],	
        initializer=tf.random_normal_initializer(stddev=config.weight_stddev),	
    )	

    if config.train_embeddings:	
        context_embed_weights = dropout(	
            context_embed_weights, config.embed_p_drop, train	
        )	
        context_weighted_avg = dropout(context_weighted_avg, config.embed_p_drop, train)	
    else:	
        context_embed_weights = tf.stop_gradient(context_embed_weights)	

    with tf.variable_scope("context_embedding"):	
        weighted_C = tf.multiply(	
            context, context_weighted_avg	
        )  # [batch_size, seq_length, context_dim] * [context_dim] = [batch_size, seq_length, context_dim], with weighted inputs	
        c_embed = tf.tensordot(	
            weighted_C, context_embed_weights, axes=[[2], [0]]	
        )  # [batch_size, seq_length, context_dim] * [context_dim, n_embed] = [batch_size, seq_length, n_embed]	
        c_embed = norm(c_embed, tf.get_variable_scope())
        # c_embed = tf.reduce_mean(c_embed, axis=1)
    featurizer_state['context'] = c_embed
    return featurizer_state
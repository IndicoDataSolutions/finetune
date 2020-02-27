import tensorflow as tf

from finetune.util.shapes import shape_list, lengths_from_eos_idx


def embed(X, we):
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h


def glove_featurizer(X, encoder, config, train=False, reuse=None, **kwargs):
    initial_shape = tf.shape(X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-2:]), 0))     
    X.set_shape([None, None, None])

    with tf.variable_scope("model/featurizer", reuse=reuse):
        embed_weights = tf.get_variable(
            name="we",
            shape=[encoder.vocab_size, config.n_embed],
            initializer=tf.random_normal_initializer(stddev=config.weight_stddev),         
        )          
        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        h = embed(X, embed_weights)

    pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], encoder.clf_token), tf.float32), 1), tf.int32)
    lengths = lengths_from_eos_idx(eos_idx=pool_idx, max_length=shape_list(X)[0])
    mask = tf.expand_dims(tf.sequence_mask(pool_idx, maxlen=tf.shape(h)[1], dtype=tf.float32), axis=-1)
    clf_h = tf.reduce_sum(h * mask, axis=1) / tf.reduce_sum(mask, axis=1)
 
    return {
        "embed_weights": embed_weights, 
        "features": clf_h,
        "sequence_features": h,
        "eos_idx": pool_idx, 
        "lengths": lengths
    }

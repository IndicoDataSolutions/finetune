import tensorflow as tf
from finetune.base_models.gpt.featurizer import dropout

def embed(X, we):
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h


def glove_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    context=None,
    context_dim=None,
    **kwargs):

    X.set_shape([None, None, None])
    input_ids = X[:, :, 0]  # slice off pos-embed ids.
    X = tf.reshape(input_ids, [-1, config.max_length, 1])

    with tf.variable_scope("model/featurizer", reuse=reuse):
        with tf.device("/CPU:0"):
            embed_weights = tf.get_variable(
                name="word_embedding",
                shape=[encoder.vocab_size, config.n_embed],
                initializer=tf.random_normal_initializer(stddev=config.weight_stddev),
            )

            if config.train_embeddings:
                embed_weights = dropout(embed_weights, config.embed_p_drop, train)
            else:
                embed_weights = tf.stop_gradient(embed_weights)

        h = embed(X, embed_weights)

        clf_h = tf.reduce_mean(h, axis=1)

        seq_feats = h

    out = {
        "embed_weights": seq_feats,
        "features": clf_h,
        "sequence_features": seq_feats,
        "pool_idx": None,
    }
    
    return out
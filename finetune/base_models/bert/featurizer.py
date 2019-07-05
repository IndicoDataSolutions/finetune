import tensorflow as tf
from finetune.base_models.bert.modeling import BertConfig, BertModel, dropout, layer_norm


def bert_featurizer(X, encoder, config, train=False, reuse=None, context=None, context_dim=None,**kwargs):
    """
    The transformer element of the finetuning model. Maps from tokens ids to a dense, embedding of the sequence.

    :param X: A tensor of token indexes with shape [batch_size, sequence_length, token_idx]
    :param encoder: A TextEncoder object.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :return: A dict containing;
        embed_weights: the word embedding matrix.
        features: The output of the featurizer_final state.
        sequence_features: The output of the featurizer at each timestep.
    """

    bert_config = BertConfig(
        vocab_size=encoder.vocab_size,
        hidden_size=config.n_embed,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_heads,
        intermediate_size=config.bert_intermediate_size,
        hidden_act=config.act_fn,
        hidden_dropout_prob=config.resid_p_drop,
        attention_probs_dropout_prob=config.attn_p_drop,
        max_position_embeddings=config.max_length,
        type_vocab_size=2,
        initializer_range=config.weight_stddev,
        adapter_size=config.adapter_size
    )

    initial_shape = tf.shape(X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-2:]), 0))
    X.set_shape([None, None, None])
    # To fit the interface of finetune we are going to compute the mask and type id at runtime.
    input_ids = X[:, :, 0]  # slice off pos-embed ids.
    delimiters = tf.cast(tf.equal(input_ids, encoder.delimiter), tf.int32)
    token_type_ids = tf.cumsum(delimiters, exclusive=True, axis=1)

    seq_length = tf.shape(delimiters)[1]

    lengths = tf.argmax(
        tf.cast(delimiters, tf.float32) *
        tf.expand_dims(tf.range(tf.cast(seq_length, tf.float32), dtype=tf.float32), 0),
        axis=1
    )

    mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)
    if config.num_layers_trained not in [config.n_layer, 0]:
        raise ValueError("Bert base model does not support num_layers_trained not equal to 0 or n_layer")

    with tf.variable_scope('model/featurizer', reuse=reuse):
        bert = BertModel(
            config=bert_config,
            is_training=train,
            input_ids=input_ids,
            input_mask=mask,
            token_type_ids=token_type_ids,
            use_one_hot_embeddings=False,
            scope=None
        )

        embed_weights = bert.get_embedding_table()
        features = tf.reshape(
                bert.get_pooled_output(),
                shape=tf.concat((initial_shape[: -2], [config.n_embed]), 0))
        sequence_features = tf.reshape(
                bert.get_sequence_output(),
                shape=tf.concat((initial_shape[:-1], [config.n_embed]), 0))

        if config.use_auxiliary_info:
            context_embed_weights = tf.get_variable(
                name="ce",
                shape=[context_dim, config.n_embed],
                initializer=tf.random_normal_initializer(stddev=config.weight_stddev))

            context_weighted_avg = tf.get_variable(
                name='cwa',
                shape=[context_dim],
                initializer=tf.random_normal_initializer(stddev=config.weight_stddev)
            )
            
            if train:
                context_embed_weights = dropout(context_embed_weights, config.embed_p_drop)

            with tf.variable_scope('context_embedding'):
                weighted_C = tf.multiply(context, context_weighted_avg) # [batch_size, seq_length, context_dim] * [context_dim] = [batch_size, seq_length, context_dim], with weighted inputs
                c_embed = tf.tensordot(weighted_C, context_embed_weights, axes = [[2],[0]]) # [batch_size, seq_length, context_dim] * [context_dim, n_embed] = [batch_size, seq_length, n_embed]
                c_embed = layer_norm(c_embed, tf.get_variable_scope())
                sequence_features = sequence_features + c_embed
                features = features + tf.reduce_sum(c_embed, axis=1)


        output_state = {
            'embed_weights': embed_weights,
            'features': features,
            'sequence_features': sequence_features,
            'pool_idx': lengths
        }
        if config.num_layers_trained == 0:
            output_state = {k: tf.stop_gradient(v) for k, v in output_state.items()}

        return output_state

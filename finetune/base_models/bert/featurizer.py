import tensorflow as tf
from finetune.base_models.bert.modelling import BertConfig, BertModel


def bert_featurizer(X, encoder, config, train=False, reuse=None):
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
        initializer_range=config.weight_stddev
    )

    # To fit the interface of finetune we are going to compute the mask and type id at runtime.
    input_ids = X[:, :, 0]  # slice of pos embed ids.
    delimeters = tf.cast(tf.equal(input_ids, encoder.delimiter), tf.int32)
    token_type_ids = tf.cumsum(delimeters, exclusive=True, axis=1)

    seq_length = tf.shape(delimeters)[1]

    lengths = tf.argmax(
        tf.cast(delimeters, tf.float32) *
        tf.expand_dims(tf.range(tf.cast(seq_length, tf.float32), dtype=tf.float32), 0),
        axis=1
    )

    mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)

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

        return {
            'embed_weights': bert.get_embedding_table(),
            'features': bert.get_pooled_output(),
            'sequence_features': bert.get_sequence_output(),
            'pool_idx': lengths
        }

import tensorflow as tf
from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder
from finetune.base_models.bert.modeling import BertConfig, BertModel


def bert_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    **kwargs
):
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

    is_roberta = RoBERTaEncoder == config.base_model.encoder

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
        adapter_size=config.adapter_size,
        low_memory_mode=config.low_memory_mode
    )

    initial_shape = tf.shape(X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-2:]), 0))
    X.set_shape([None, None, None])
    # To fit the interface of finetune we are going to compute the mask and type id at runtime.
    input_ids = X[:, :, 0]  # slice off pos-embed ids.
    delimiters = tf.cast(tf.equal(input_ids, encoder.delimiter_token), tf.int32)

    token_type_ids = tf.cumsum(delimiters, exclusive=True, axis=1)

    seq_length = tf.shape(delimiters)[1]

    eos_idx = tf.argmax(
        tf.cast(delimiters, tf.float32)
        * tf.expand_dims(
            tf.range(tf.cast(seq_length, tf.float32), dtype=tf.float32), 0
        ),
        axis=1,
    )

    lengths = lengths_from_eos_idx(eos_idx=eos_idx, max_length=seq_length)

    if is_roberta:
        # Because roberta embeddings include an unused <MASK> token, and our embedding
        #  layer size needs to accommodate for that.
        bert_config.vocab_size += 1
        # In our use case (padding token has index 1), roberta's position indexes begin at 2, so our
        # positions embeddings come from indices 2:514.
        bert_config.max_position_embeddings += 2

    mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)

    if config.num_layers_trained not in [config.n_layer, 0]:
        raise ValueError(
            "Bert base model does not support num_layers_trained not equal to 0 or n_layer"
        )

    with tf.variable_scope("model/featurizer", reuse=reuse):
        bert = BertModel(
            config=bert_config,
            is_training=train,
            input_ids=input_ids,
            input_mask=mask,
            token_type_ids=token_type_ids,
            use_one_hot_embeddings=False,
            scope=None,
            use_pooler=config.bert_use_pooler,
            use_token_type=config.bert_use_type_embed,
            roberta=is_roberta
        )

        embed_weights = bert.get_embedding_table()
        features = tf.reshape(
            bert.get_pooled_output(),
            shape=tf.concat((initial_shape[:-2], [config.n_embed]), 0),
        )
        sequence_features = tf.reshape(
            bert.get_sequence_output(),
            shape=tf.concat((initial_shape[:-1], [config.n_embed]), 0),
        )

        output_state = {
            "embed_weights": embed_weights,
            "features": features,
            "sequence_features": sequence_features,
            "lengths": lengths,
            "eos_idx": eos_idx,
        }
        if config.num_layers_trained == 0:
            output_state = {k: tf.stop_gradient(v) for k, v in output_state.items()}

        return output_state

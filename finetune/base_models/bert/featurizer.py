import functools

import tensorflow as tf
from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder
from finetune.base_models.bert.modeling import (
    BertConfig,
    BertModel,
    LayoutLMModel,
    XDocModel,
    TwinBertModel,
)
from finetune.base_models.bert.table_utils import (
    get_gather_indices,
    get_row_col_values,
    table_cross_row_col_mixing_fn,
    reassemble_sequence_feats,
)


def get_decay_for_half(total_num_steps):
    decay = tf.minimum(
        tf.cast(tf.compat.v1.train.get_global_step(), tf.float32)
        / (total_num_steps / 2),
        1.0,
    )
    tf.compat.v1.summary.scalar("positional_decay_rate", decay)
    return decay


def bert_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    context=None,
    total_num_steps=None,
    underlying_model=BertModel,
    max_length=None,
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

    is_roberta = config.base_model.is_roberta
    model_filename = config.base_model_path.rpartition("/")[-1]
    is_roberta_v1 = is_roberta and config.base_model.encoder == RoBERTaEncoder

    if max_length is None:
        max_length = config.max_length
    bert_config = BertConfig(
        vocab_size=encoder.vocab_size,
        hidden_size=config.n_embed,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_heads,
        intermediate_size=config.bert_intermediate_size,
        hidden_act=config.act_fn,
        hidden_dropout_prob=config.resid_p_drop,
        attention_probs_dropout_prob=config.attn_p_drop,
        max_position_embeddings=max_length,
        type_vocab_size=2,
        initializer_range=config.weight_stddev,
        low_memory_mode=config.low_memory_mode,
        pos_injection=config.context_injection,
        reading_order_removed=config.reading_order_removed,
        anneal_reading_order=config.anneal_reading_order,
        positional_channels=config.context_channels,
    )

    initial_shape = tf.shape(input=X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))
    X.set_shape([None, None])
    # To fit the interface of finetune we are going to compute the mask and type id at runtime.
    delimiters = tf.cast(tf.equal(X, encoder.delimiter_token), tf.int32)
    token_type_ids = tf.cumsum(delimiters, exclusive=True, axis=1)

    seq_length = tf.shape(input=delimiters)[1]

    def get_eos():
        return tf.argmax(
            input=tf.cast(delimiters, tf.float32)
            * tf.expand_dims(
                tf.range(tf.cast(seq_length, tf.float32), dtype=tf.float32), 0
            ),
            axis=1,
        )

    def get_zeros():
        return tf.zeros(tf.shape(X)[0], dtype=tf.int64)

    eos_idx = tf.cond(tf.equal(seq_length, 0), true_fn=get_zeros, false_fn=get_eos)

    lengths = lengths_from_eos_idx(eos_idx=eos_idx, max_length=seq_length)

    if is_roberta:
        # In our use case (padding token has index 1), roberta's position indexes begin at 2, so our
        # positions embeddings come from indices 2:514.
        if is_roberta_v1:
            # v1 vocab didn't include MASK token although the embedding did
            bert_config.vocab_size += 1

        bert_config.max_position_embeddings += 2

    mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)

    if config.num_layers_trained not in [config.n_layer, 0]:
        raise ValueError(
            "Bert base model does not support num_layers_trained not equal to 0 or n_layer"
        )

    if config.anneal_reading_order:
        reading_order_decay_rate = get_decay_for_half(total_num_steps)
    else:
        reading_order_decay_rate = None

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        bert = underlying_model(
            config=bert_config,
            is_training=train,
            input_ids=X,
            input_mask=mask,
            input_context=context,
            token_type_ids=token_type_ids,
            use_one_hot_embeddings=False,
            scope=None,
            use_pooler=config.bert_use_pooler,
            use_token_type=config.bert_use_type_embed,
            roberta=is_roberta,
            reading_order_decay_rate=reading_order_decay_rate,
        )

        embed_weights = bert.get_embedding_table()
        features = tf.reshape(
            bert.get_pooled_output(),
            shape=tf.concat((initial_shape[:-1], [config.n_embed]), 0),
        )
        sequence_features = tf.reshape(
            bert.get_sequence_output(layer_num=config.feature_layer_num),
            shape=tf.concat((initial_shape, [config.n_embed]), 0),
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


layoutlm_featurizer = functools.partial(bert_featurizer, underlying_model=LayoutLMModel)
xdoc_featurizer = functools.partial(bert_featurizer, underlying_model=XDocModel)


def table_roberta_featurizer_twinbert(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    context=None,
    underlying_model=TwinBertModel,
    max_length=None,
    lengths=None,
    **kwargs
):
    """
    The backbone of the table model.

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
    is_roberta = config.base_model.is_roberta
    is_roberta_v1 = is_roberta and config.base_model.encoder == RoBERTaEncoder

    if max_length is None:
        max_length = config.max_length
    bert_config = BertConfig(
        vocab_size=encoder.vocab_size,
        hidden_size=config.n_embed,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_heads,
        intermediate_size=config.bert_intermediate_size,
        hidden_act=config.act_fn,
        hidden_dropout_prob=config.resid_p_drop,
        attention_probs_dropout_prob=config.attn_p_drop,
        max_position_embeddings=512,  # Overriden because the max length of the transformers is not the max length from our config.
        type_vocab_size=2,
        initializer_range=config.weight_stddev,
        low_memory_mode=config.low_memory_mode,
        pos_injection=config.context_injection,
        reading_order_removed=config.reading_order_removed,
        anneal_reading_order=config.anneal_reading_order,
        positional_channels=config.context_channels,
        table_position=config.table_position,
        table_position_type=config.table_position_type,
        max_row_col_embedding=config.max_row_col_embedding,
    )

    initial_shape = tf.shape(input=X)
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))

    if is_roberta:
        # In our use case (padding token has index 1), roberta's position indexes begin at 2, so our
        # positions embeddings come from indices 2:514.
        if is_roberta_v1:
            # v1 vocab didn't include MASK token although the embedding did
            bert_config.vocab_size += 1

        bert_config.max_position_embeddings += 2

    if config.num_layers_trained not in [config.n_layer, 0]:
        raise ValueError(
            "Bert base model does not support num_layers_trained not equal to 0 or n_layer"
        )

    output_shape = tf.concat([tf.shape(X), [config.n_embed]], axis=0)

    end_col, end_row, start_col, start_row = tf.unstack(context, num=4, axis=2)
    row_gather = get_gather_indices(X, lengths, start_row, end_row, other_end=end_col, chunk_tables=config.chunk_tables)
    col_gather = get_gather_indices(X, lengths, start_col, end_col, other_end=end_row, chunk_tables=config.chunk_tables)

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        row_col_values = get_row_col_values(
            X,
            context,
            row_gather,
            col_gather,
            bos_id=encoder.start_token,
            eos_id=encoder.end_token,
            table_position_type=config.table_position_type,
            max_row_col_embedding=config.max_row_col_embedding
        )
        bert = underlying_model(
            mixing_fn=table_cross_row_col_mixing_fn,
            mixing_inputs=dict(
                output_shape=output_shape,
                row_col_values=row_col_values,
                row_gather=row_gather,
                col_gather=col_gather,
            ),
            config=bert_config,
            is_training=train,
            input_ids_a=row_col_values["row"]["values"],
            input_ids_b=row_col_values["col"]["values"],
            attention_mask_a=row_col_values["row"]["attn_mask"],
            attention_mask_b=row_col_values["col"]["attn_mask"],
            token_type_ids_a=tf.zeros_like(row_col_values["row"]["values"]),
            token_type_ids_b=tf.zeros_like(row_col_values["col"]["values"]),
            context_a=row_col_values["row"]["positions"],
            context_b=row_col_values["col"]["positions"],
            pos_ids_a=row_gather["pos_ids"],
            pos_ids_b=col_gather["pos_ids"],
            use_one_hot_embeddings=False,
            scope=None,
            use_token_type=config.bert_use_type_embed,
            roberta=is_roberta,
            reading_order_decay_rate=None,
        )

        output_state = {
            "embed_weights": bert.get_embedding_table(),
            "features": tf.zeros(shape=[initial_shape[0], 768]),
            "sequence_features": reassemble_sequence_feats(
                output_shape,
                *bert.sequence_output, #row, col
                row_col_values["row"]["scatter_vals"],
                row_col_values["col"]["scatter_vals"],
                include_row_col_summaries=config.include_row_col_summaries,
                down_project_feats=config.down_project_feats,
            ),
            "lengths": lengths,
            "eos_idx": tf.zeros(tf.shape(X)[0], dtype=tf.int64),
        }
        if config.num_layers_trained == 0:
            output_state = {k: tf.stop_gradient(v) for k, v in output_state.items()}

        return output_state

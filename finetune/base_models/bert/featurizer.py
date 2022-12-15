import os
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
    gelu
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

MAX_ROWS_AND_COLS = 50

def slice_and_dice_tables(
    X, sequence_lengths, scatter_idx_orig, start, end, other_start, max_rows_cols, bos_id, eos_id, max_length=512
):
    col_values_list = []
    col_scatter_list = []
    mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(X)[1])
    for i in range(max_rows_cols):
        i = float(i)
        # SHould not include pad values as these will have a max/min row/col of -1
        col_mask = tf.math.logical_and(
            tf.math.logical_and(
                tf.math.less_equal(start, i), tf.math.less_equal(i, end)
            ),
            mask,
        )
        #col_mask = tf.compat.v1.Print(col_mask, [col_mask])
        col_values_i = tf.ragged.boolean_mask(X, col_mask)
        col_scatter_i = tf.ragged.boolean_mask(scatter_idx_orig, col_mask)
        # Which batches have non-0 number of entries in row/col X
        batch_mask = tf.math.not_equal(col_values_i.row_lengths(), 0)

        col_values = tf.ragged.boolean_mask(col_values_i, batch_mask)
        col_bs = tf.shape(col_values.row_lengths())[0]
        bos_expanded = tf.tile(tf.RaggedTensor.from_tensor(tf.expand_dims(tf.expand_dims(bos_id, 0), 0)), [col_bs, 1, *(1 for _ in bos_id.shape)])
        eos_expanded = tf.tile(tf.RaggedTensor.from_tensor(tf.expand_dims(tf.expand_dims(eos_id, 0), 0)), [col_bs, 1, *(1 for _ in bos_id.shape)])
        col_values = tf.concat(
            [
                bos_expanded,
                col_values,
                eos_expanded,
            ],
            1
        )
        col_scatter = tf.ragged.boolean_mask(col_scatter_i, batch_mask)
        col_scatter = tf.concat(
            [
                tf.tile(
                    tf.ragged.constant(
                        [[[-1, -1]]]
                    ),
                    [col_bs, 1, 1]
                ),
                col_scatter,
                tf.tile(
                    tf.ragged.constant(
                        [[[-1, -1]]]
                    ),
                    [col_bs, 1, 1]
                ),
            ],
            1
        )
        col_values_list.append(col_values)
        col_scatter_list.append(col_scatter)
    col_values_ragged = tf.concat(col_values_list, axis=0)
    col_scatter_ragged = tf.concat(col_scatter_list, axis=0)
    col_seq_lens = col_values_ragged.row_lengths()
    # I don't think this default matters here as these are masked. Cannot be < 0
    col_values = col_values_ragged.to_tensor(
        default_value=1234, shape=col_values_ragged.bounding_shape()
    )
    col_scatter = col_scatter_ragged.to_tensor(
        default_value=-1, shape=col_scatter_ragged.bounding_shape()
    )
    with tf.control_dependencies(
        [
            tf.cond(
                tf.shape(col_scatter)[1] > max_length,
                false_fn=tf.no_op,
                true_fn=lambda: tf.compat.v1.Print(
                    True,
                    [
                        "The length of produced tensors is > {} shape is: ".format(
                            max_length
                        ),
                        tf.shape(col_scatter),
                    ],
                ),
            )
        ]
    ):
        col_values = col_values[:, :max_length]
        col_scatter = col_scatter[:, :max_length]
    return {
        "seq_lens": col_seq_lens,
        "values": col_values,
        "scatter_vals": col_scatter,
    }


def get_row_col_values(X, context, sequence_lengths, bos_id, eos_id, max_rows_cols=MAX_ROWS_AND_COLS):
    #context = tf.compat.v1.Print(context, [context], summarize=10000)
    end_col, end_row, start_col, start_row = tf.unstack(context, num=4, axis=2)
    batch_idx = tf.tile(
        tf.expand_dims(tf.range(tf.shape(X)[0]), 1), [1, tf.shape(X)[1]]
    )
    seq_idx = tf.tile(tf.expand_dims(tf.range(tf.shape(X)[1]), 0), [tf.shape(X)[0], 1])
    scatter_idx_orig = tf.stack([batch_idx, seq_idx], axis=-1)  # batch, seq, 2
    return {
        "row": slice_and_dice_tables(
            X, sequence_lengths, scatter_idx_orig, start_row, end_row, start_col, max_rows_cols, bos_id=tf.constant(bos_id), eos_id=tf.constant(eos_id)
        ),
        "col": slice_and_dice_tables(
            X, sequence_lengths, scatter_idx_orig, start_col, end_col, start_row, max_rows_cols, bos_id=tf.constant(bos_id), eos_id=tf.constant(eos_id)
        ),
    }


def scatter_feats(output_shape, sequence_feats, scatter_vals):
    input_tensor = tf.zeros(shape=output_shape, dtype=tf.float32)
    input_tensor.set_shape([None, None, 768])
    mask = tf.math.not_equal(scatter_vals[:, :, 0], -1)
    feats = tf.boolean_mask(sequence_feats, mask)
    scatter_idxs = tf.boolean_mask(scatter_vals, mask)
    # Averages any cases where tokens are in multiple cells - for example when cells span multiple rows / cols.
    divide_by = tf.tensor_scatter_nd_add(
        tf.zeros_like(input_tensor), scatter_idxs, tf.ones_like(feats)
    )
    #divide_by = tf.compat.v1.Print(divide_by, ["Divide by", divide_by[0, :, 0]], summarize=100000)
    return tf.math.divide_no_nan(
        tf.tensor_scatter_nd_add(
            input_tensor, scatter_idxs, feats
        ),
        divide_by
    )


def reassemble_sequence_feats(
    shape, row_sequence_feats, col_sequence_feats, row_scatter_vals, col_scatter_vals
):
    return tf.concat(
        [
            scatter_feats(shape, col_sequence_feats, col_scatter_vals),
            scatter_feats(shape, row_sequence_feats, row_scatter_vals),
        ],
        -1,
    )

def adaptor_block(inp, hidden_dim):
    # Note - no residual in here as the residual connection will be from the original stack and not the mixed stack.
    hidden = tf.compat.v1.layers.dense(inp, hidden_dim, activation=gelu, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1e-3))
    return tf.compat.v1.layers.dense(hidden, inp.shape[-1], activation=None, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1e-3))

def table_cross_row_col_mixing_fn(row_feats, col_feats, *, lengths, context, output_shape, row_col_values, max_rows_cols=MAX_ROWS_AND_COLS):
    bos_var = tf.compat.v1.get_variable(
        'bos',
        shape=[768],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(),
        trainable=True
    )
    eos_var = tf.compat.v1.get_variable(
        'eos',
        shape=[768],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(),
        trainable=True
    )
    end_col, end_row, start_col, start_row = tf.unstack(context, num=4, axis=2)
    batch_idx = tf.tile(
        tf.expand_dims(tf.range(output_shape[0]), 1), [1, output_shape[1]]
    )
    seq_idx = tf.tile(tf.expand_dims(tf.range(output_shape[1]), 0), [output_shape[0], 1])
    scatter_idx_orig = tf.stack([batch_idx, seq_idx], axis=-1)  # batch, seq, 2

    col_feats_orig_shape = scatter_feats(output_shape, col_feats, row_col_values["col"]["scatter_vals"])
    row_feats_orig_shape = scatter_feats(output_shape, row_feats, row_col_values["row"]["scatter_vals"])

    # Scatter col feats into rows arrangement
    col_feats_reshaped = slice_and_dice_tables(
        col_feats_orig_shape, lengths, scatter_idx_orig, start_row, end_row, start_col, max_rows_cols, bos_id=bos_var, eos_id=eos_var
    )["values"]

    # Scatter row feats into cols arrangement.
    row_feats_reshaped = slice_and_dice_tables(
        row_feats_orig_shape, lengths, scatter_idx_orig, start_col, end_col, start_row, max_rows_cols, bos_id=bos_var, eos_id=eos_var
    )["values"]

    return adaptor_block(col_feats_reshaped, 64) + row_feats, adaptor_block(row_feats_reshaped, 64) + col_feats


    

def table_roberta_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    context=None,
    total_num_steps=None,
    underlying_model=BertModel,
    max_length=None,
    lengths=None,
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
        max_position_embeddings=512,  # Overriden because we have a much longer effective max length than the models support.
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
    # X.set_shape([None, None])
    # To fit the interface of finetune we are going to compute the mask and type id at runtime.
    delimiters = tf.cast(tf.equal(X, encoder.delimiter_token), tf.int32)
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

    if config.anneal_reading_order:
        reading_order_decay_rate = get_decay_for_half(total_num_steps)
    else:
        reading_order_decay_rate = None

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        row_col_values = get_row_col_values(X, context, lengths, bos_id=encoder.start_token, eos_id=encoder.end_token)
        with tf.compat.v1.variable_scope("row"):
            bert = underlying_model(
                config=bert_config,
                is_training=train,
                input_ids=row_col_values["row"]["values"],
                input_mask=tf.sequence_mask(
                    row_col_values["row"]["seq_lens"],
                    maxlen=tf.shape(row_col_values["row"]["values"])[1],
                    dtype=tf.float32,
                ),
                input_context=None,
                token_type_ids=tf.zeros_like(row_col_values["row"]["values"]),
                use_one_hot_embeddings=False,
                scope=None,
                use_pooler=config.bert_use_pooler,
                use_token_type=config.bert_use_type_embed,
                roberta=is_roberta,
                reading_order_decay_rate=reading_order_decay_rate,
            )
            embed_weights = bert.get_embedding_table()
            row_sequence_features = bert.get_sequence_output(
                layer_num=config.feature_layer_num
            )
        with tf.compat.v1.variable_scope("col"):
            bert = underlying_model(
                config=bert_config,
                is_training=train,
                input_ids=row_col_values["col"]["values"],
                input_mask=tf.sequence_mask(
                    row_col_values["col"]["seq_lens"],
                    maxlen=tf.shape(row_col_values["col"]["values"])[1],
                    dtype=tf.float32,
                ),
                input_context=None,
                token_type_ids=tf.zeros_like(row_col_values["col"]["values"]),
                use_one_hot_embeddings=False,
                scope=None,
                use_pooler=config.bert_use_pooler,
                use_token_type=config.bert_use_type_embed,
                roberta=is_roberta,
                reading_order_decay_rate=reading_order_decay_rate,
            )
            embed_weights = bert.get_embedding_table()
            col_sequence_features = bert.get_sequence_output(
                layer_num=config.feature_layer_num
            )
        #row_sequence_features = tf.compat.v1.Print(row_sequence_features, ["row shape", tf.shape(row_sequence_features), "col shape", tf.shape(col_sequence_features)])
        output_state = {
            "embed_weights": embed_weights,
            "features": tf.zeros(shape=[initial_shape[0], 768]),
            "sequence_features": reassemble_sequence_feats(
                tf.concat([tf.shape(X), [config.n_embed]], axis=0),
                row_sequence_features,
                col_sequence_features,
                row_col_values["row"]["scatter_vals"],
                row_col_values["col"]["scatter_vals"],
            ),
            "lengths": lengths,
            "eos_idx": eos_idx,
        }
        if config.num_layers_trained == 0:
            output_state = {k: tf.stop_gradient(v) for k, v in output_state.items()}

        return output_state



def table_roberta_featurizer_twinbert(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    context=None,
    total_num_steps=None,
    underlying_model=TwinBertModel,
    max_length=None,
    lengths=None,
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
        max_position_embeddings=512,  # Overriden because we have a much longer effective max length than the models support.
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
    # X.set_shape([None, None])
    # To fit the interface of finetune we are going to compute the mask and type id at runtime.
    delimiters = tf.cast(tf.equal(X, encoder.delimiter_token), tf.int32)
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

    if config.anneal_reading_order:
        reading_order_decay_rate = get_decay_for_half(total_num_steps)
    else:
        reading_order_decay_rate = None
    output_shape = tf.concat([tf.shape(X), [config.n_embed]], axis=0)
    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        row_col_values = get_row_col_values(X, context, lengths, bos_id=encoder.start_token, eos_id=encoder.end_token)
        bert = underlying_model(
            mixing_fn=table_cross_row_col_mixing_fn,
            mixing_inputs=dict(
                lengths=lengths,
                context=context,
                output_shape=output_shape,
                row_col_values=row_col_values,
            ),
            config=bert_config,
            is_training=train,
            input_ids_a=row_col_values["row"]["values"],
            input_ids_b=row_col_values["col"]["values"],

            input_mask_a=tf.sequence_mask(
                row_col_values["row"]["seq_lens"],
                maxlen=tf.shape(row_col_values["row"]["values"])[1],
                dtype=tf.float32,
            ),
            input_mask_b=tf.sequence_mask(
                row_col_values["col"]["seq_lens"],
                maxlen=tf.shape(row_col_values["col"]["values"])[1],
                dtype=tf.float32,
            ),
            token_type_ids_a=tf.zeros_like(row_col_values["row"]["values"]),
            token_type_ids_b=tf.zeros_like(row_col_values["col"]["values"]),            
            use_one_hot_embeddings=False,
            scope=None,
            use_token_type=config.bert_use_type_embed,
            roberta=is_roberta,
            reading_order_decay_rate=reading_order_decay_rate,
        )
        embed_weights = bert.get_embedding_table()
        row_sequence_features, col_sequence_features = bert.sequence_output

        #row_sequence_features = tf.compat.v1.Print(row_sequence_features, ["row shape", tf.shape(row_sequence_features), "col shape", tf.shape(col_sequence_features)])
        output_state = {
            "embed_weights": embed_weights,
            "features": tf.zeros(shape=[initial_shape[0], 768]),
            "sequence_features": reassemble_sequence_feats(
                output_shape,
                row_sequence_features,
                col_sequence_features,
                row_col_values["row"]["scatter_vals"],
                row_col_values["col"]["scatter_vals"],
            ),
            "lengths": lengths,
            "eos_idx": eos_idx,
        }
        if config.num_layers_trained == 0:
            output_state = {k: tf.stop_gradient(v) for k, v in output_state.items()}

        return output_state

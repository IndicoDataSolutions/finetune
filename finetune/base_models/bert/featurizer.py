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
    gelu,
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


def get_col_masks(X, sequence_lengths, start, end):
    mask = tf.expand_dims(
        tf.sequence_mask(sequence_lengths, maxlen=tf.shape(X)[1]), 0
    )  # 1, batch, seq
    range = tf.expand_dims(
        tf.expand_dims(tf.range(tf.reduce_max(end) + 1, dtype=start.dtype), 1), 1
    )  # num_cols, 1, 1
    start = tf.expand_dims(start, 0)  # 1, batch, max_len
    end = tf.expand_dims(end, 0)  # 1, batch, max_len
    return tf.math.logical_and(
        tf.math.logical_and(
            tf.math.less_equal(start, range), tf.math.less_equal(range, end)
        ),
        mask,
    )  # num_cols, batch, seq


@tf.function(autograph=False)
def batch_packing(ragged_input, include_mask=True):
    # Basically the next fit binpacking algorithm.

    # If there is some fast way to sort the rows of a ragged tensor we might consider doing this first.
    col_seq_lens = ragged_input.row_lengths()
    max_length = tf.reduce_max(col_seq_lens)

    def loop_body(l, i):
        temp_ragged = tf.RaggedTensor.from_row_lengths(
            values=col_seq_lens[:i],
            row_lengths=l,
        )
        return (
            tf.cond(
                pred=tf.math.less_equal(
                    tf.minimum(tf.reduce_sum(temp_ragged[-1]), 512),
                    max_length - col_seq_lens[i],
                ),
                true_fn=lambda: tf.concat((l[:-1], [l[-1] + 1]), axis=0),
                false_fn=lambda: tf.concat((l, [1]), axis=0),
            ),
            i + 1,
        )

    lengths, _ = tf.while_loop(
        cond=lambda _, i: tf.math.less(i, tf.shape(col_seq_lens)[0]),
        body=loop_body,
        loop_vars=(tf.ones([1], dtype=tf.int64), tf.constant(1)),
        parallel_iterations=1,
        shape_invariants=(tf.TensorShape([None]), tf.TensorShape([])),
    )

    lengths = tf.RaggedTensor.from_row_lengths(
        values=col_seq_lens,
        row_lengths=lengths,
    )
    ragged = tf.RaggedTensor.from_row_lengths(
        values=ragged_input.flat_values, row_lengths=tf.reduce_sum(lengths, 1)
    )
    if include_mask:
        lengths_ts = tf.math.cumsum(lengths.to_tensor(), 1)
        raw_mask = tf.sequence_mask(lengths_ts, dtype=tf.float32)
        compound_mask = tf.reduce_sum(
            tf.expand_dims(raw_mask, 2) * tf.expand_dims(raw_mask, 3), 1
        )
        row_max = tf.reduce_max(compound_mask, 1)
        output_mask = tf.cast(
            tf.logical_and(
                tf.equal(compound_mask, tf.expand_dims(row_max, 1)),
                tf.equal(compound_mask, tf.expand_dims(row_max, 2)),
            ),
            tf.float32,
        )
    else:
        output_mask = None
    # output_mask = tf.compat.v1.Print(output_mask, ["Original", col_seq_lens, "After packing", ragged.row_lengths()], summarize=100000)
    return ragged, output_mask


@tf.function(autograph=False)
def _slice_and_dice_single(
    inp,
    col_masks,
    eos_pad,
    bos_pad,
    pad_val,
    max_length=512,
    max_tokens_per_batch=512 * 100,
    check_len=False,
    include_mask=False,
):
    print("Running slice and dice single")
    bos_pad_ragged = tf.RaggedTensor.from_tensor(
        tf.expand_dims(tf.expand_dims(bos_pad, 0), 0)
    )
    eos_pad_ragged = tf.RaggedTensor.from_tensor(
        tf.expand_dims(tf.expand_dims(eos_pad, 0), 0)
    )

    # def map_fn_internal(col_mask):
    #     inp_values_i = tf.ragged.boolean_mask(inp, col_mask)
    #     batch_mask = tf.math.not_equal(inp_values_i.row_lengths(), 0)
    #     inp_values = tf.RaggedTensor.from_row_lengths(
    #         values=inp_values_i.flat_values,
    #         row_lengths=tf.boolean_mask(inp_values_i.row_lengths(), batch_mask)
    #     )
    #     col_bs = tf.shape(inp_values.row_lengths())[0]
    #     bos_expanded = tf.tile(bos_pad_ragged, [col_bs, 1, *(1 for _ in bos_pad.shape)])
    #     eos_expanded = tf.tile(eos_pad_ragged, [col_bs, 1, *(1 for _ in eos_pad.shape)])
    #     inp_values = tf.concat(
    #         [
    #             bos_expanded,
    #             inp_values,
    #             eos_expanded,
    #         ],
    #         axis=1,
    #     )
    #     return inp_values

    # output_ragged = tf.map_fn(
    #     map_fn_internal,
    #     col_masks,
    #     fn_output_signature=tf.RaggedTensorSpec(
    #         shape=[None, None] + inp.shape[2:],
    #         dtype=inp.dtype,
    #         ragged_rank=1,
    #     ),
    #     parallel_iterations=1,
    # ).merge_dims(0, 1)

    inp_expanded = tf.expand_dims(inp, 0)  # 1, bs, seq, ...
    inp_values_i = tf.ragged.boolean_mask(
        tf.tile(inp_expanded, [tf.shape(col_masks)[0], 1, 1, *(1 for _ in bos_pad.shape)]),
        col_masks,
    ).merge_dims(0, 1)
    batch_mask = tf.math.not_equal(inp_values_i.row_lengths(), 0)
    inp_values = tf.RaggedTensor.from_row_lengths(
           values=inp_values_i.flat_values,
           row_lengths=tf.boolean_mask(inp_values.row_lengths(), batch_mask)
    )
    col_bs = tf.shape(inp_values.row_lengths())[0]
    bos_expanded = tf.tile(bos_pad_ragged, [col_bs, 1, *(1 for _ in bos_pad.shape)])
    eos_expanded = tf.tile(eos_pad_ragged, [col_bs, 1, *(1 for _ in eos_pad.shape)])
    output_ragged = tf.concat(
        [
            bos_expanded,
            inp_values,
            eos_expanded,
        ],
        axis=1,
    )
    output_ragged, mask = batch_packing(output_ragged, include_mask=include_mask)
    col_seq_lens = output_ragged.row_lengths()

    # I don't think this default matters here as these are masked. Cannot be < 0
    col_values = output_ragged.to_tensor(
        default_value=pad_val, shape=[None, None] + inp.shape[2:]
    )
    # This is to handle some edge cases where 0 length values are missing the final dim after conversion.
    col_values.set_shape([None, None] + list(inp.shape[2:]))
    max_length = tf.minimum(
        tf.math.floordiv(max_tokens_per_batch, tf.shape(col_values)[0]), max_length
    )
    if check_len:
        ctrl_dep = [
            tf.cond(
                tf.shape(col_values)[1] > max_length,
                false_fn=tf.no_op,
                true_fn=lambda: tf.compat.v1.Print(
                    True,
                    [
                        "The length of produced tensors is >",
                        max_length,
                        "shape is: ",
                        tf.shape(col_values),
                    ],
                ),
            )
        ]
    else:
        ctrl_dep = []
    with tf.control_dependencies(ctrl_dep):
        col_values = col_values[:, :max_length]
        if include_mask:
            mask = mask[:, :max_length, :max_length]
            mask.set_shape([None, None, None])

    col_values.set_shape([None, None] + list(inp.shape[2:]))
    return {
        "seq_lens": col_seq_lens,  # tf.compat.v1.Print(col_seq_lens, ["Lengths", col_seq_lens], summarize=1000),
        "values": col_values,
        "attn_mask": mask,
    }

def slice_and_dice_single(inp, *args, **kwargs):
    with tf.device("cpu"):
        output = _slice_and_dice_single(inp, *args, **kwargs)
    output["values"].set_shape([None, None] + list(inp.shape[2:]))
    return output

def slice_and_dice_tables(
    X,
    col_masks,
    scatter_idx_orig,
    context,
    bos_id,
    eos_id,
    table_position_type,
):
    if table_position_type == "row_col":
        pos_raw = context[:, :, 2:]
        # 1023 because it's least likely to be used so can be a special token for EOS and BOS.
        pos_eos = tf.constant([1023, 1023])
    elif table_position_type == "all":
        pos_raw = context
        pos_eos = tf.constant([1023, 1023, 1023, 1023])

    return {
        **slice_and_dice_single(
            X,
            col_masks,
            eos_id,
            bos_id,
            pad_val=1234,
            check_len=True,
            include_mask=True,
        ),
        "scatter_vals": slice_and_dice_single(
            scatter_idx_orig,
            col_masks,
            tf.constant([-1, -1]),
            tf.constant([-1, -1]),
            pad_val=-1,
        )["values"],
        "positions": slice_and_dice_single(
            tf.cast(pos_raw, tf.int32), col_masks, pos_eos, pos_eos, pad_val=1
        )["values"],
    }


def get_row_col_values(
    X, context, row_masks, col_masks, bos_id, eos_id, table_position_type
):
    batch_idx = tf.tile(
        tf.expand_dims(tf.range(tf.shape(X)[0]), 1), [1, tf.shape(X)[1]]
    )
    seq_idx = tf.tile(tf.expand_dims(tf.range(tf.shape(X)[1]), 0), [tf.shape(X)[0], 1])
    scatter_idx_orig = tf.stack([batch_idx, seq_idx], axis=-1)  # batch, seq, 2
    scatter_idx_orig = tf.reshape(scatter_idx_orig, [tf.shape(X)[0], tf.shape(X)[1], 2])
    return {
        "row": slice_and_dice_tables(
            X,
            row_masks,
            scatter_idx_orig,
            context,
            bos_id=tf.constant(bos_id),
            eos_id=tf.constant(eos_id),
            table_position_type=table_position_type,
        ),
        "col": slice_and_dice_tables(
            X,
            col_masks,
            scatter_idx_orig,
            context,
            bos_id=tf.constant(bos_id),
            eos_id=tf.constant(eos_id),
            table_position_type=table_position_type,
        ),
    }


@tf.function(autograph=False)
def _scatter_feats(output_shape, sequence_feats, scatter_vals):
    print("Running scatter feats")
    input_tensor = tf.zeros(shape=output_shape, dtype=tf.float32)
    # input_tensor.set_shape([None, None, 768])
    # scatter_vals.set_shape([None, None, 2])
    mask = tf.math.not_equal(scatter_vals[:, :, 0], -1)
    feats = tf.boolean_mask(sequence_feats, mask)
    scatter_idxs = tf.boolean_mask(scatter_vals, mask)
    # Averages any cases where tokens are in multiple cells - for example when cells span multiple rows / cols.
    divide_by = tf.tensor_scatter_nd_add(
        input_tensor, scatter_idxs, tf.ones_like(feats)
    )
    # divide_by = tf.compat.v1.Print(divide_by, ["Divide by", divide_by[0, :, 0]], summarize=100000)
    return tf.math.divide_no_nan(
        tf.tensor_scatter_nd_add(input_tensor, scatter_idxs, feats), divide_by
    )

def scatter_feats(*args, **kwargs):
    output = _scatter_feats(*args, **kwargs)
    output.set_shape([None, None, 768])
    return output

def reassemble_sequence_feats(
    shape,
    row_sequence_feats,
    col_sequence_feats,
    row_scatter_vals,
    col_scatter_vals,
    include_row_col_summaries,
    down_project_feats,
):
    if include_row_col_summaries:
        feat_dim = 768 * 4
        summaries = [
            scatter_feats(
                shape,
                tf.tile(
                    col_sequence_feats[:, :1, :],
                    [1, tf.shape(col_sequence_feats)[1], 1],
                ),
                col_scatter_vals,
            ),
            scatter_feats(
                shape,
                tf.tile(
                    row_sequence_feats[:, :1, :],
                    [1, tf.shape(row_sequence_feats)[1], 1],
                ),
                row_scatter_vals,
            ),
        ]
    else:
        feat_dim = 768 * 2
        summaries = []

    feats = tf.concat(
        [
            scatter_feats(shape, col_sequence_feats, col_scatter_vals),
            scatter_feats(shape, row_sequence_feats, row_scatter_vals),
            *summaries,
        ],
        -1,
    )
    feats.set_shape([None, None, feat_dim])
    if down_project_feats:
        feats = tf.compat.v1.layers.dense(feats, 768, activation=gelu)

    return feats


def adaptor_block(inp, hidden_dim):
    # Note - no residual in here as the residual connection will be from the original stack and not the mixed stack.
    hidden = tf.compat.v1.layers.dense(
        inp,
        hidden_dim,
        activation=gelu,
        kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1e-3),
    )
    return tf.compat.v1.layers.dense(
        hidden,
        inp.shape[-1],
        activation=None,
        kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1e-3),
    )


def table_cross_row_col_mixing_fn(
    row_feats, col_feats, *, row_masks, col_masks, output_shape, row_col_values
):
    bos_var = tf.compat.v1.get_variable(
        "bos",
        shape=[768],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(),
        trainable=True,
    )
    eos_var = tf.compat.v1.get_variable(
        "eos",
        shape=[768],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(),
        trainable=True,
    )

    col_feats_orig_shape = scatter_feats(
        output_shape, col_feats, row_col_values["col"]["scatter_vals"]
    )
    row_feats_orig_shape = scatter_feats(
        output_shape, row_feats, row_col_values["row"]["scatter_vals"]
    )

    # Scatter col feats into rows arrangement
    col_feats_reshaped = slice_and_dice_single(
        col_feats_orig_shape, row_masks, bos_pad=bos_var, eos_pad=eos_var, pad_val=1234
    )["values"]

    # Scatter row feats into cols arrangement.
    row_feats_reshaped = slice_and_dice_single(
        row_feats_orig_shape, col_masks, bos_pad=bos_var, eos_pad=eos_var, pad_val=1234
    )["values"]

    return (
        adaptor_block(col_feats_reshaped, 64) + row_feats,
        adaptor_block(row_feats_reshaped, 64) + col_feats,
    )


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

    end_col, end_row, start_col, start_row = tf.unstack(context, num=4, axis=2)
    row_masks = get_col_masks(X, lengths, start_row, end_row)
    col_masks = get_col_masks(X, lengths, start_col, end_col)

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        row_col_values = get_row_col_values(
            X,
            context,
            row_masks,
            col_masks,
            bos_id=encoder.start_token,
            eos_id=encoder.end_token,
            table_position_type="row_col",
        )
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
        # row_sequence_features = tf.compat.v1.Print(row_sequence_features, ["row shape", tf.shape(row_sequence_features), "col shape", tf.shape(col_sequence_features)])
        output_state = {
            "embed_weights": embed_weights,
            "features": tf.zeros(shape=[initial_shape[0], 768]),
            "sequence_features": reassemble_sequence_feats(
                tf.concat([tf.shape(X), [config.n_embed]], axis=0),
                row_sequence_features,
                col_sequence_features,
                row_col_values["row"]["scatter_vals"],
                row_col_values["col"]["scatter_vals"],
                include_row_col_summaries=config.include_row_col_summaries,
                down_project_feats=config.down_project_feats,
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
        table_position=config.table_position,
        table_position_type=config.table_position_type,
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
    row_masks = get_col_masks(X, lengths, start_row, end_row)
    col_masks = get_col_masks(X, lengths, start_col, end_col)

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        row_col_values = get_row_col_values(
            X,
            context,
            row_masks,
            col_masks,
            bos_id=encoder.start_token,
            eos_id=encoder.end_token,
            table_position_type=config.table_position_type,
        )
        bert = underlying_model(
            mixing_fn=table_cross_row_col_mixing_fn,
            mixing_inputs=dict(
                output_shape=output_shape,
                row_col_values=row_col_values,
                row_masks=row_masks,
                col_masks=col_masks,
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
            use_one_hot_embeddings=False,
            scope=None,
            use_token_type=config.bert_use_type_embed,
            roberta=is_roberta,
            reading_order_decay_rate=None,
        )
        embed_weights = bert.get_embedding_table()
        row_sequence_features, col_sequence_features = bert.sequence_output

        # row_sequence_features = tf.compat.v1.Print(row_sequence_features, ["row shape", tf.shape(row_sequence_features), "col shape", tf.shape(col_sequence_features)])
        output_state = {
            "embed_weights": embed_weights,
            "features": tf.zeros(shape=[initial_shape[0], 768]),
            "sequence_features": reassemble_sequence_feats(
                output_shape,
                row_sequence_features,
                col_sequence_features,
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

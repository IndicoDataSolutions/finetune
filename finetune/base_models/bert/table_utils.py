import tensorflow as tf
from finetune.base_models.bert.modeling import gelu


def get_gather_indices(X, sequence_lengths, start, end, other_end, chunk_tables):
    """
    Given the tokens, sequence lengths and the range of each token in either rows or columns
    gets the values required for gathering the input tokens into row/column stack inputs and
    scattering the features back into it's original shape.

    It also returns the masks to be used by the attention operations and a new set of
    sequence lengths.

    Internally - Special tokens are appended to X before the gathers are applied.

    Additionally, long rows / columns are chunked up, retaining 2 columns / rows of context (intended to be the headers) into every chunk.

    Args:
        X: Input tokens [batch, sequence]
        sequence_lengths: Number of tokens per sequence [batch]
        start: The start of the col/row range.
        end: The end of the col/row range.
        other_end: The end of the row/col range. This is used to determine the first 2 rows / cols to retain in all chunks for context.
        chunk_tables: a boolean. Whether to apply the chunking operation described above 

    Returns:
    A dict:
    {
        "seq_lens": New sequence lengths [new_batch],
        "values": Indexes into the original sequence of the new batches of data. [new_batch, new_seq_len],
        "attn_mask": Mask for the new sequences [new_batch, new_seq_len, new_seq_len],
    }

    """
    seq_len = tf.shape(X)[1]
    bs = tf.shape(X)[0]

    mask = tf.expand_dims(
        tf.sequence_mask(sequence_lengths, maxlen=seq_len), 0
    )  # 1, batch, seq
    range = tf.expand_dims(
        tf.expand_dims(tf.range(tf.reduce_max(end) + 1, dtype=start.dtype), 1), 1
    )  # num_cols, 1, 1
    start = tf.expand_dims(start, 0)  # 1, batch, max_len
    end = tf.expand_dims(end, 0)  # 1, batch, max_len
    col_masks = tf.math.logical_and(
        tf.math.logical_and(
            tf.math.less_equal(start, range), tf.math.less_equal(range, end)
        ),
        mask,
    )  # num_cols, batch, seq

    batch_idx = tf.tile(tf.expand_dims(tf.range(bs), 1), [1, seq_len])
    seq_idx = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [bs, 1])
    scatter_idx_orig = tf.stack([batch_idx, seq_idx], axis=-1)  # batch, seq, 2
    scatter_idx_orig = tf.reshape(scatter_idx_orig, [bs, seq_len, 2])
    return slice_by_table_indices(
        scatter_idx_orig,
        col_masks,
        tf.convert_to_tensor([0, seq_len]),
        tf.convert_to_tensor([0, seq_len + 1]),
        pad_val=tf.convert_to_tensor([0, seq_len + 2]),
        other_end=other_end,
        include_mask=True,
        chunk_tables=chunk_tables,
    )


def batch_packing(ragged_input, include_mask=True, base_model_max_length=512):
    """
    Takes a ragged tensor input and re-packs the batches to minimise the batch size without
    impacting the sequence length. Additionally returns a mask to use with self-attention so 
    that there is no change in output.

    Args:
        ragged_input: A ragged tensor of at least rank 2. - The inputs to repack.
        include_mask (bool, optional): Whether to return the masks.

    Returns:
        new output - repacked but still ragged.
        the mask [batch, seq_len, seq_len]
        position ids [batch, seq_len]
    """
    # Basically the next fit binpacking algorithm.

    # If there is some fast way to sort the rows of a ragged tensor we might consider doing this first.
    col_seq_lens = ragged_input.row_lengths()
    max_length = tf.minimum(tf.reduce_max(col_seq_lens), base_model_max_length)

    def loop_body(l, i):
        temp_ragged = tf.RaggedTensor.from_row_lengths(
            values=col_seq_lens[:i], row_lengths=l
        )
        return (
            tf.cond(
                pred=tf.math.less_equal(
                    tf.minimum(tf.reduce_sum(temp_ragged[-1]), base_model_max_length),
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

    lengths = tf.RaggedTensor.from_row_lengths(values=col_seq_lens, row_lengths=lengths)
    ragged = tf.RaggedTensor.from_row_lengths(
        values=ragged_input.flat_values, row_lengths=tf.reduce_sum(lengths, 1)
    )
    if include_mask:
        lengths_ts = tf.math.cumsum(lengths.to_tensor(), 1)
        raw_mask = tf.sequence_mask(
            lengths_ts,
            dtype=tf.float32,
            maxlen=tf.minimum(base_model_max_length, tf.cast(tf.reduce_max(lengths_ts), tf.int32)),
        )
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

    pos_ids = tf.RaggedTensor.from_row_lengths(
        values=tf.ragged.range(lengths.flat_values).flat_values,
        row_lengths=tf.reduce_sum(lengths, 1),
    )
    return ragged, output_mask, pos_ids


def chunk_ragged_tensor(
    inputs: tf.RaggedTensor,
    other_end: tf.Tensor,
    include_n_rows: int = 2,
    base_model_max_length: int = 512- 2,
     # -2 to account for BOS and EOS tokens that will be added.
):
    """
    Chunks up inputs, keeping the first include_n_rows in every chunk.
    """
    batch_size = inputs.bounding_shape()[0]

    # Figure out which tokens are in the first include_n_rows rows of the column and split these off.

    is_first_n_rows_orig = tf.gather_nd(
        tf.math.less(other_end, include_n_rows), inputs, batch_dims=0
    )
    less_than_max_length_tokens = tf.math.less(
        tf.math.reduce_sum(tf.cast(is_first_n_rows_orig, tf.int64), axis=1), base_model_max_length
    )
    # If first row tokens is more than base_model_max_length - do not use any first row context.
    is_first_n_rows = tf.logical_and(
        tf.expand_dims(less_than_max_length_tokens, 1), is_first_n_rows_orig
    )
    num_first_n_row_tokens = tf.math.reduce_sum(
        tf.cast(is_first_n_rows, tf.int64), axis=1
    )

    num_cols_per_col = tf.cast(
        tf.maximum(
            tf.math.ceil(
                (inputs.row_lengths() - num_first_n_row_tokens)
                / (base_model_max_length - num_first_n_row_tokens)
            ),
            1,  # Max(1 .) is to catch the case where all tokens are in the first n rows.
        ),
        tf.int64,
    )
    first_n_rows = tf.ragged.boolean_mask(inputs, is_first_n_rows)
    other_rows = tf.ragged.boolean_mask(inputs, tf.math.logical_not(is_first_n_rows))
    other_rows_length_goals = base_model_max_length - num_first_n_row_tokens  # bs

    # Reshape the rest of the columns to the per-column length_goal
    new_row_breaks = tf.cast(
        tf.minimum(
            tf.ragged.range(
                (num_cols_per_col + 1) * other_rows_length_goals,
                deltas=other_rows_length_goals,
                dtype=tf.int64,
            ),
            tf.expand_dims(other_rows.row_lengths(), 1),
        ),
        tf.int64,
    )
    new_row_lengths = new_row_breaks[:, 1:] - new_row_breaks[:, :-1]
    other_rows_reshaped = tf.RaggedTensor.from_row_lengths(
        values=other_rows.flat_values, row_lengths=new_row_lengths.flat_values
    )

    # Tile the first N rows so that they are duplicated for each chunk of the column

    # a really gross way to get range broadcasted to num_cols_per_col - not sure of a better way to get this.
    indexes = tf.ragged.range(num_cols_per_col) * 0 + tf.expand_dims(
        tf.range(batch_size), 1
    )
    first_n_rows_duped = tf.gather(first_n_rows, indexes).merge_dims(0, 1)

    result = tf.concat([first_n_rows_duped, other_rows_reshaped], 1)
    return result


def slice_by_table_indices(
    inp,
    col_masks,
    eos_pad,
    bos_pad,
    pad_val,
    other_end,
    base_model_max_length=512,
    max_tokens_per_batch=512 * 100,
    check_len=False,
    include_mask=False,
    chunk_tables=True,
):
    """
    Used internally by get_gather_indices - assumes that any pre-processing of adding special tokens is already complete.
    Can handle higher rank inputs but it seems to be quicker to just generate the indices and then gather.
    """
    with tf.device("cpu"):
        bos_pad_ragged = tf.RaggedTensor.from_tensor(
            tf.expand_dims(tf.expand_dims(bos_pad, 0), 0)
        )
        eos_pad_ragged = tf.RaggedTensor.from_tensor(
            tf.expand_dims(tf.expand_dims(eos_pad, 0), 0)
        )

        inp_expanded = tf.expand_dims(inp, 0)  # 1, bs, seq, ...
        inp_values_i = tf.ragged.boolean_mask(
            tf.tile(
                inp_expanded,
                [tf.shape(col_masks)[0], 1, 1, *(1 for _ in bos_pad.shape)],
            ),
            col_masks,
        ).merge_dims(0, 1)
        batch_mask = tf.math.not_equal(inp_values_i.row_lengths(), 0)
        inp_values = tf.RaggedTensor.from_row_lengths(
            values=inp_values_i.flat_values,
            row_lengths=tf.boolean_mask(inp_values_i.row_lengths(), batch_mask),
        )
        if chunk_tables:
            inp_values = chunk_ragged_tensor(
                inp_values, other_end=other_end,
                base_model_max_length=base_model_max_length - 2
            )  # -2 for eos and bos
        col_bs = tf.shape(inp_values.row_lengths())[0]
        bos_expanded = tf.tile(bos_pad_ragged, [col_bs, 1, *(1 for _ in bos_pad.shape)])
        eos_expanded = tf.tile(eos_pad_ragged, [col_bs, 1, *(1 for _ in eos_pad.shape)])
        output_ragged = tf.concat([bos_expanded, inp_values, eos_expanded], axis=1)

        output_ragged, mask, pos_ids = batch_packing(
            output_ragged, include_mask=include_mask, base_model_max_length=base_model_max_length
        )
        col_seq_lens = output_ragged.row_lengths()

        # I don't think this default matters here as these are masked. Cannot be < 0
        col_values = output_ragged.to_tensor(
            default_value=pad_val, shape=[None, None] + inp.shape[2:]
        )
        pos_ids = pos_ids.to_tensor(default_value=0, shape=[None, None])
    # This is to handle some edge cases where 0 length values are missing the final dim after conversion.
    col_values.set_shape([None, None] + list(inp.shape[2:]))
    max_length = tf.minimum(
        tf.math.floordiv(max_tokens_per_batch, tf.shape(col_values)[0]), base_model_max_length
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
    if not chunk_tables:
        with tf.control_dependencies(ctrl_dep):
            col_values = col_values[:, :max_length]
            if include_mask:
                mask = mask[:, :max_length, :max_length]
                mask.set_shape([None, None, None])
            pos_ids = pos_ids[:, :max_length]

    col_values.set_shape([None, None] + list(inp.shape[2:]))
    return {
        "seq_lens": col_seq_lens,
        "values": col_values,
        "attn_mask": mask,
        "pos_ids": pos_ids,
    }


def gather_col_vals(inp, gather_output, eos_pad, bos_pad, pad_val):
    """Used with the output of get_gather_indices to gather the input to the models.

    Args:
        inp: input to be gathered at least rank 2 with the first 2 dimensions [batch, sequence]_
        gather_output: THe output of get_gather_indices
        eos_pad: A value to be used as the end of sequence value - equal to the rank of inp - 2.
        bos_pad: A value to be used as the beginning of sequence value - equal to the rank of inp - 2.
        pad_val A value to be used as the padding value - equal to the rank of inp - 2.

    Returns:
        a dictionary 
        {
            "seq_lens": seq_lens from gather output unmodified,
            "values": the values inp rearraged to be ready as input to the model,
            "attn_mask": mask from gather output unmodified,
        }
    """
    bs = tf.shape(inp)[0]
    bos_expanded = tf.tile(
        tf.expand_dims(tf.expand_dims(bos_pad, 0), 0),
        [bs, 1, *(1 for _ in bos_pad.shape)],
    )
    eos_expanded = tf.tile(
        tf.expand_dims(tf.expand_dims(eos_pad, 0), 0),
        [bs, 1, *(1 for _ in eos_pad.shape)],
    )
    inp_w_extra_toks = tf.concat(
        [inp, bos_expanded, eos_expanded, tf.ones_like(eos_expanded) * pad_val], axis=1
    )
    values = tf.gather_nd(indices=gather_output["values"], params=inp_w_extra_toks)
    return {
        "seq_lens": gather_output["seq_lens"],
        "values": values,
        "attn_mask": gather_output["attn_mask"],
    }


def gather_tables(
    X, col_gather, context, bos_id, eos_id, table_position_type, max_row_col_embedding
):
    """
    Prepares all the inputs required for initially feeding to one half of the table model.

    Args:
        X: Input tokens [batch, seq]
        col_gather: the output of get_gather_indices
        context: context (usually start and end of each row and column) to be gathered and passed to the model.
        bos_id: The BOS token id. (scalar)
        eos_id: _The EOS token id. (scalar)
        table_position_type: One of "row_col" or "all" - the type of position embedding used.

    Returns:
        _type_: _description_
    """
    pad_id = max_row_col_embedding - 1
    if table_position_type == "row_col":
        pos_raw = context[:, :, 2:]
        pos_eos = tf.constant([pad_id, pad_id])
    elif table_position_type == "all":
        pos_raw = context
        pos_eos = tf.constant([pad_id, pad_id, pad_id, pad_id])

    return {
        **gather_col_vals(X, col_gather, eos_id, bos_id, pad_val=1234),
        "scatter_vals": col_gather["values"],
        "positions": gather_col_vals(
            tf.cast(pos_raw, tf.int32), col_gather, pos_eos, pos_eos, pad_val=1
        )["values"],
    }


def get_row_col_values(
    X,
    context,
    row_gather,
    col_gather,
    bos_id,
    eos_id,
    table_position_type,
    max_row_col_embedding,
):
    """
    Applys gather_tables on rows and columns and returns the output as a nested dict
    with keys "row" and "col"
    """
    return {
        "row": gather_tables(
            X,
            row_gather,
            context,
            bos_id=tf.constant(bos_id),
            eos_id=tf.constant(eos_id),
            table_position_type=table_position_type,
            max_row_col_embedding=max_row_col_embedding,
        ),
        "col": gather_tables(
            X,
            col_gather,
            context,
            bos_id=tf.constant(bos_id),
            eos_id=tf.constant(eos_id),
            table_position_type=table_position_type,
            max_row_col_embedding=max_row_col_embedding,
        ),
    }


def scatter_feats(output_shape, sequence_feats, scatter_vals):
    """
    The inverse of gather_col_vals - maps from columnwise inputs back to the original layout.
    When features are represented in multiple places mean reduction is used.
    0 is used as the default where no tokens in the input map to tokens in the output.

    Args:
        output_shape: Expected shape of the output.
        sequence_feats: features from either the row or col model.
        scatter_vals: the scatter vals output from gather_tables - the indices of the original positions of each token.

    Returns:
        The reformatted features.
    """
    input_tensor = tf.zeros(shape=output_shape, dtype=tf.float32)
    mask = tf.math.less(
        scatter_vals[:, :, 0], output_shape[1]
    )  # Special tokens were placed after the length of the shape.
    feats = tf.boolean_mask(sequence_feats, mask)
    scatter_idxs = tf.boolean_mask(scatter_vals, mask)
    # Averages any cases where tokens are in multiple cells - for example when cells span multiple rows / cols.
    divide_by = tf.tensor_scatter_nd_add(
        input_tensor, scatter_idxs, tf.ones_like(feats)
    )
    return tf.math.divide_no_nan(
        tf.tensor_scatter_nd_add(input_tensor, scatter_idxs, feats), divide_by
    )


def get_summary_values(inp, gather_vals, input_seq_len):
    bos_mask = tf.reduce_all(
        tf.equal(tf.convert_to_tensor([0, input_seq_len + 1]), gather_vals), 2
    )
    bos_mask_idxs = tf.range(tf.shape(inp)[1]) * tf.cast(bos_mask, tf.int32)
    bos_mask_idxs_t = tf.transpose(bos_mask_idxs, [1, 0])
    cumulative_max = tf.scan(
        lambda a, b: tf.maximum(a, b),
        bos_mask_idxs_t,
        initializer=tf.reduce_min(bos_mask_idxs_t, axis=0),
    )
    return tf.gather(inp, tf.transpose(cumulative_max, [1, 0]), batch_dims=1)


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
                get_summary_values(col_sequence_feats, col_scatter_vals, shape[1]),
                col_scatter_vals,
            ),
            scatter_feats(
                shape,
                get_summary_values(row_sequence_feats, row_scatter_vals, shape[1]),
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
    row_feats, col_feats, *, row_gather, col_gather, output_shape, row_col_values
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
    col_feats_reshaped = gather_col_vals(
        col_feats_orig_shape, row_gather, bos_pad=bos_var, eos_pad=eos_var, pad_val=1234
    )["values"]

    # Scatter row feats into cols arrangement.
    row_feats_reshaped = gather_col_vals(
        row_feats_orig_shape, col_gather, bos_pad=bos_var, eos_pad=eos_var, pad_val=1234
    )["values"]

    return (
        adaptor_block(col_feats_reshaped, 64) + row_feats,
        adaptor_block(row_feats_reshaped, 64) + col_feats,
    )

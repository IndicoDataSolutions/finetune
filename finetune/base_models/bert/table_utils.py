import logging

import numpy as np
import tensorflow as tf

from finetune.nn.activations import bert_gelu as gelu

LOGGER = logging.getLogger("finetune")


def _split_ragged_rows_numpy(values, row_lengths):
    rows = []
    offset = 0
    for row_length in row_lengths:
        next_offset = offset + int(row_length)
        rows.append(values[offset:next_offset])
        offset = next_offset
    return rows


def _flatten_ragged_rows_numpy(rows):
    if not rows:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.int64)
    row_lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    non_empty_rows = [row for row in rows if len(row)]
    if non_empty_rows:
        values = np.concatenate(non_empty_rows, axis=0).astype(np.int32, copy=False)
    else:
        values = np.zeros((0, 2), dtype=np.int32)
    return values, row_lengths


def _masked_ragged_indices_numpy_rows(sequence_lengths, start, end):
    bs = int(sequence_lengths.shape[0])
    per_batch_groups = []
    global_max_axis = -1

    for batch_idx in range(bs):
        seq_len = int(sequence_lengths[batch_idx])
        batch_groups = []
        for seq_idx in range(seq_len):
            start_idx = int(start[batch_idx, seq_idx])
            end_idx = int(end[batch_idx, seq_idx])
            if end_idx >= len(batch_groups):
                batch_groups.extend([] for _ in range(end_idx + 1 - len(batch_groups)))
            token_idx = (batch_idx, seq_idx)
            for axis_idx in range(start_idx, end_idx + 1):
                batch_groups[axis_idx].append(token_idx)
        per_batch_groups.append(batch_groups)
        global_max_axis = max(global_max_axis, len(batch_groups) - 1)

    rows = []
    for axis_idx in range(global_max_axis + 1):
        for batch_idx in range(bs):
            batch_groups = per_batch_groups[batch_idx]
            if axis_idx >= len(batch_groups):
                continue
            axis_values = batch_groups[axis_idx]
            if not axis_values:
                continue
            rows.append(np.asarray(axis_values, dtype=np.int32))

    return rows


def _masked_ragged_indices_numpy(sequence_lengths, start, end):
    rows = _masked_ragged_indices_numpy_rows(sequence_lengths, start, end)
    values, row_lengths = _flatten_ragged_rows_numpy(rows)

    if len(row_lengths) == 0:
        return tf.RaggedTensor.from_row_lengths(
            values=tf.zeros([0, 2], dtype=tf.int32),
            row_lengths=tf.zeros([0], dtype=tf.int64),
        )

    return tf.RaggedTensor.from_row_lengths(
        values=tf.convert_to_tensor(values, dtype=tf.int32),
        row_lengths=tf.convert_to_tensor(row_lengths, dtype=tf.int64),
    )


def _chunk_rows_numpy(
    rows,
    other_end,
    include_n_rows=2,
    base_model_max_length=512 - 2,
):
    chunked_rows = []
    first_n_mask = np.asarray(other_end < include_n_rows, dtype=bool)

    for row in rows:
        if len(row) == 0:
            chunked_rows.append(row)
            continue
        row_mask = first_n_mask[row[:, 0], row[:, 1]]
        if np.count_nonzero(row_mask) >= base_model_max_length:
            row_mask = np.zeros_like(row_mask, dtype=bool)
        first_rows = row[row_mask]
        other_rows = row[~row_mask]
        remaining_budget = base_model_max_length - len(first_rows)
        if remaining_budget <= 0:
            remaining_budget = base_model_max_length

        if len(other_rows) == 0:
            chunked_rows.append(first_rows)
            continue

        for start_idx in range(0, len(other_rows), remaining_budget):
            chunked_rows.append(
                np.concatenate(
                    [first_rows, other_rows[start_idx : start_idx + remaining_budget]],
                    axis=0,
                )
            )

    return chunked_rows


def _build_block_diagonal_mask_and_pos_ids(row_lengths):
    total_length = int(np.sum(row_lengths))
    if total_length == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    mask = np.zeros((total_length, total_length), dtype=np.float32)
    pos_ids = np.zeros((total_length,), dtype=np.int32)
    offset = 0
    for row_length in row_lengths:
        next_offset = offset + int(row_length)
        mask[offset:next_offset, offset:next_offset] = 1.0
        pos_ids[offset:next_offset] = np.arange(row_length, dtype=np.int32)
        offset = next_offset
    return mask, pos_ids


def _batch_packed_rows_numpy(
    rows,
    include_mask=True,
    base_model_max_length=512,
):
    if not rows:
        empty_mask = np.zeros((0, 0, 0), dtype=np.float32) if include_mask else None
        return (
            [],
            np.zeros((0,), dtype=np.int32),
            empty_mask,
            np.zeros((0, 0), dtype=np.int32),
        )

    row_lengths = np.asarray([len(row) for row in rows], dtype=np.int32)
    max_length = min(int(np.max(row_lengths)), base_model_max_length)

    groups = [[0]]
    current_total = int(row_lengths[0])
    for row_idx in range(1, len(rows)):
        next_length = int(row_lengths[row_idx])
        if min(current_total, base_model_max_length) <= max_length - next_length:
            groups[-1].append(row_idx)
            current_total += next_length
        else:
            groups.append([row_idx])
            current_total = next_length

    packed_rows = []
    packed_seq_lens = []
    packed_pos_ids = []
    packed_masks = []

    for group in groups:
        group_rows = [rows[row_idx] for row_idx in group]
        packed_rows.append(np.concatenate(group_rows, axis=0))
        group_row_lengths = [len(group_row) for group_row in group_rows]
        packed_seq_lens.append(sum(group_row_lengths))
        mask, pos_ids = _build_block_diagonal_mask_and_pos_ids(group_row_lengths)
        packed_pos_ids.append(pos_ids)
        if include_mask:
            packed_masks.append(mask)

    max_seq_len = max((len(row) for row in packed_rows), default=0)
    dense_pos_ids = np.zeros((len(packed_rows), max_seq_len), dtype=np.int32)
    dense_masks = (
        np.zeros((len(packed_rows), max_seq_len, max_seq_len), dtype=np.float32)
        if include_mask
        else None
    )

    for idx, pos_ids in enumerate(packed_pos_ids):
        dense_pos_ids[idx, : len(pos_ids)] = pos_ids
        if include_mask:
            dense_masks[idx, : len(pos_ids), : len(pos_ids)] = packed_masks[idx]
            dense_masks[idx, len(pos_ids) :, len(pos_ids) :] = 1.0

    return (
        packed_rows,
        np.asarray(packed_seq_lens, dtype=np.int32),
        dense_masks,
        dense_pos_ids,
    )


def _build_gather_outputs_inference_numpy_arrays(
    X,
    sequence_lengths,
    start,
    end,
    other_end,
    chunk_tables,
    include_mask,
    base_model_max_length=512,
    max_tokens_per_batch=512 * 100,
):
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int32)
    start = np.asarray(start, dtype=np.int32)
    end = np.asarray(end, dtype=np.int32)
    other_end = np.asarray(other_end, dtype=np.int32)
    X = np.asarray(X, dtype=np.int32)

    rows = _masked_ragged_indices_numpy_rows(
        sequence_lengths=sequence_lengths, start=start, end=end
    )
    if chunk_tables:
        rows = _chunk_rows_numpy(
            rows,
            other_end=other_end,
            base_model_max_length=base_model_max_length - 2,
        )

    seq_len = int(X.shape[1])
    bos_pad = np.asarray([[0, seq_len + 1]], dtype=np.int32)
    eos_pad = np.asarray([[0, seq_len]], dtype=np.int32)
    rows = [
        np.concatenate([bos_pad, row, eos_pad], axis=0)
        if len(row)
        else np.concatenate([bos_pad, eos_pad], axis=0)
        for row in rows
    ]

    packed_rows, seq_lens, attn_mask, pos_ids = _batch_packed_rows_numpy(
        rows,
        include_mask=include_mask,
        base_model_max_length=base_model_max_length,
    )

    max_seq_len = max((len(row) for row in packed_rows), default=0)
    col_values = np.full(
        (len(packed_rows), max_seq_len, 2), [0, seq_len + 2], dtype=np.int32
    )
    for idx, row in enumerate(packed_rows):
        col_values[idx, : len(row)] = row

    max_length = min(
        max_tokens_per_batch // max(len(packed_rows), 1), base_model_max_length
    )
    if not chunk_tables:
        col_values = col_values[:, :max_length]
        pos_ids = pos_ids[:, :max_length]
        if include_mask:
            attn_mask = attn_mask[:, :max_length, :max_length]

    if attn_mask is None:
        attn_mask = None

    return {
        "seq_lens": seq_lens.astype(np.int32, copy=False),
        "values": col_values.astype(np.int32, copy=False),
        "attn_mask": None
        if attn_mask is None
        else attn_mask.astype(np.float32, copy=False),
        "pos_ids": pos_ids.astype(np.int32, copy=False),
    }


def _build_gather_outputs_inference_numpy(
    X,
    sequence_lengths,
    start,
    end,
    other_end,
    chunk_tables,
    include_mask,
    base_model_max_length=512,
    max_tokens_per_batch=512 * 100,
):
    arrays = _build_gather_outputs_inference_numpy_arrays(
        X=np.asarray(X.numpy(), dtype=np.int32),
        sequence_lengths=np.asarray(sequence_lengths.numpy(), dtype=np.int32),
        start=np.asarray(start.numpy(), dtype=np.int32),
        end=np.asarray(end.numpy(), dtype=np.int32),
        other_end=np.asarray(other_end.numpy(), dtype=np.int32),
        chunk_tables=chunk_tables,
        include_mask=include_mask,
        base_model_max_length=base_model_max_length,
        max_tokens_per_batch=max_tokens_per_batch,
    )
    col_values = tf.convert_to_tensor(arrays["values"], dtype=tf.int32)
    col_values.set_shape([None, None, 2])
    return {
        "seq_lens": tf.convert_to_tensor(arrays["seq_lens"], dtype=tf.int32),
        "values": col_values,
        "attn_mask": None
        if arrays["attn_mask"] is None
        else tf.convert_to_tensor(arrays["attn_mask"], dtype=tf.float32),
        "pos_ids": tf.convert_to_tensor(arrays["pos_ids"], dtype=tf.int32),
    }


def batch_packing(
    ragged_input,
    include_mask=True,
    base_model_max_length=512,
    training=False,
    target_seq_len=None,
):
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
    if training and target_seq_len is not None:
        max_length = tf.cast(target_seq_len, tf.int64)
    else:
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
            maxlen=tf.minimum(
                base_model_max_length, tf.cast(tf.reduce_max(lengths_ts), tf.int32)
            ),
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
    base_model_max_length: int = 512 - 2,
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
        tf.math.reduce_sum(tf.cast(is_first_n_rows_orig, tf.int64), axis=1),
        base_model_max_length,
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


def compute_masked_ragged_indices(
    X,
    sequence_lengths,
    start,
    end,
    other_end,
    chunk_tables,
    base_model_max_length=512,
):
    """Build masked ragged indices for table rows/cols with optional chunking."""
    with tf.device("cpu"):
        bs = tf.shape(X)[0]
        seq_len = tf.shape(X)[1]
        mask = tf.expand_dims(
            tf.sequence_mask(sequence_lengths, maxlen=seq_len), 0
        )  # 1, batch, seq
        range_ = tf.expand_dims(
            tf.expand_dims(tf.range(tf.reduce_max(end) + 1, dtype=start.dtype), 1), 1
        )  # num_cols, 1, 1
        start_e = tf.expand_dims(start, 0)  # 1, batch, max_len
        end_e = tf.expand_dims(end, 0)  # 1, batch, max_len
        col_masks = tf.math.logical_and(
            tf.math.logical_and(
                tf.math.less_equal(start_e, range_), tf.math.less_equal(range_, end_e)
            ),
            mask,
        )  # num_cols, batch, seq

        batch_idx = tf.tile(tf.expand_dims(tf.range(bs), 1), [1, seq_len])
        seq_idx = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [bs, 1])
        scatter_idx_orig = tf.stack([batch_idx, seq_idx], axis=-1)  # batch, seq, 2

        inp_expanded = tf.expand_dims(scatter_idx_orig, 0)  # 1, bs, seq, 2

        inp_values_i = tf.ragged.boolean_mask(
            tf.tile(inp_expanded, [tf.shape(col_masks)[0], 1, 1, 1]), col_masks
        ).merge_dims(0, 1)

        batch_mask = tf.math.not_equal(inp_values_i.row_lengths(), 0)
        inp_values = tf.RaggedTensor.from_row_lengths(
            values=inp_values_i.flat_values,
            row_lengths=tf.boolean_mask(inp_values_i.row_lengths(), batch_mask),
        )
        if chunk_tables:
            inp_values = chunk_ragged_tensor(
                inp_values,
                other_end=other_end,
                base_model_max_length=base_model_max_length - 2,
            )  # -2 for EOS and BOS
            # Prepare BOS/EOS paddings
        bos_pad = tf.convert_to_tensor([0, seq_len + 1])
        eos_pad = tf.convert_to_tensor([0, seq_len])

        bos_pad_ragged = tf.RaggedTensor.from_tensor(
            tf.expand_dims(tf.expand_dims(bos_pad, 0), 0)
        )
        eos_pad_ragged = tf.RaggedTensor.from_tensor(
            tf.expand_dims(tf.expand_dims(eos_pad, 0), 0)
        )

        col_bs = tf.shape(inp_values.row_lengths())[0]
        bos_expanded = tf.tile(bos_pad_ragged, [col_bs, 1, 1])
        eos_expanded = tf.tile(eos_pad_ragged, [col_bs, 1, 1])
        output_ragged = tf.concat([bos_expanded, inp_values, eos_expanded], axis=1)
    return output_ragged


def build_gather_outputs(
    X,
    sequence_lengths,
    start,
    end,
    other_end,
    chunk_tables,
    include_mask,
    training,
    target_seq_len=None,
    target_batch_size=None,
    base_model_max_length=512,
    max_tokens_per_batch=512 * 100,
    check_len=False,
):
    with tf.device("cpu"):
        output_ragged = compute_masked_ragged_indices(
            X,
            sequence_lengths,
            start,
            end,
            other_end,
            chunk_tables,
            base_model_max_length,
        )
        pad_val = tf.convert_to_tensor([0, tf.shape(X)[1] + 2])
        output_ragged, mask, pos_ids = batch_packing(
            output_ragged,
            include_mask=include_mask,
            base_model_max_length=base_model_max_length,
            training=training,
            target_seq_len=target_seq_len,
        )
        col_seq_lens = output_ragged.row_lengths()

        # Convert to dense tensor of shape [None, None, 2]
        col_values = output_ragged.to_tensor(
            default_value=pad_val, shape=[None, None, 2]
        )
        pos_ids = pos_ids.to_tensor(default_value=0, shape=[None, None])

    # Crop if not chunking
    max_length = tf.minimum(
        tf.math.floordiv(max_tokens_per_batch, tf.shape(col_values)[0]),
        base_model_max_length,
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

    # Casts and shapes
    col_seq_lens = tf.cast(col_seq_lens, tf.int32)
    pos_ids = tf.cast(pos_ids, tf.int32)

    # Final adjustment to exact target sequence length when training
    if training and (target_seq_len is not None or target_batch_size is not None):
        seq_padding = tf.maximum(target_seq_len - tf.shape(col_values)[1], 0)
        batch_padding = tf.maximum(target_batch_size - tf.shape(col_values)[0], 0)
        pad0 = tf.pad(
            col_values[:target_batch_size, :target_seq_len, 0],
            [[0, batch_padding], [0, seq_padding]],
            constant_values=pad_val[0],
        )
        pad1 = tf.pad(
            col_values[:target_batch_size, :target_seq_len, 1],
            [[0, batch_padding], [0, seq_padding]],
            constant_values=pad_val[1],
        )
        # The ensure shapes will raise an error if the shapes don't match. But we are being
        # very defensive here so it should be fine and variations will make train time / memory explode
        # if we get this even slightly wrong.
        col_values = tf.ensure_shape(
            tf.stack([pad0, pad1], axis=-1),
            [target_batch_size, target_seq_len, 2],
        )
        if include_mask:
            mask = tf.ensure_shape(
                tf.pad(
                    mask[:target_batch_size, :target_seq_len, :target_seq_len],
                    [[0, batch_padding], [0, seq_padding], [0, seq_padding]],
                    constant_values=0,
                ),
                [target_batch_size, target_seq_len, target_seq_len],
            )
        pos_ids = tf.ensure_shape(
            tf.pad(
                pos_ids[:target_batch_size, :target_seq_len],
                [[0, batch_padding], [0, seq_padding]],
                constant_values=0,
            ),
            [target_batch_size, target_seq_len],
        )
        col_seq_lens = tf.ensure_shape(
            tf.pad(
                col_seq_lens[:target_batch_size],
                [[0, batch_padding]],
                constant_values=0,
            ),
            [target_batch_size],
        )

    col_values.set_shape([None, None, 2])
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
    hidden_dim = inp.shape[2:]
    batch_size = tf.shape(inp)[0]
    token_bcast_shape = [batch_size, 1, *hidden_dim]
    bos_expanded = tf.broadcast_to(bos_pad, token_bcast_shape)
    eos_expanded = tf.broadcast_to(eos_pad, token_bcast_shape)
    pad_expanded = tf.broadcast_to(pad_val, token_bcast_shape)
    inp_w_extra_toks = tf.concat(
        [inp, bos_expanded, eos_expanded, pad_expanded], axis=1
    )
    values = tf.gather_nd(indices=gather_output["values"], params=inp_w_extra_toks)
    # output always has the same rank as the input.
    values = tf.ensure_shape(values, [None for _ in inp.shape])
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
            bos_id=tf.convert_to_tensor(bos_id),
            eos_id=tf.convert_to_tensor(eos_id),
            table_position_type=table_position_type,
            max_row_col_embedding=max_row_col_embedding,
        ),
        "col": gather_tables(
            X,
            col_gather,
            context,
            bos_id=tf.convert_to_tensor(bos_id),
            eos_id=tf.convert_to_tensor(eos_id),
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
        output_shape: Expected shape of the output, representing (text_batch, text_seq, feat_dim)
        sequence_feats: features from either the row or col model. Shape (table_batch, table_seq, feat_dim)
        scatter_vals: the scatter vals output from gather_tables - the indices of the original positions of each token. (table_batch, table_seq, 2)

    Returns:
        The reformatted features.
    """
    input_tensor = tf.zeros(
        shape=output_shape, dtype=tf.float32
    )  # [text_batch, text_seq, feat_dim]
    mask = tf.math.less(
        scatter_vals[:, :, 1], output_shape[1]
    )  # Mask any tokens mapped outside of the input_tensor shape [text_batch, text_seq]
    # Special tokens were placed after the length of the shape.
    feats = tf.boolean_mask(sequence_feats, mask)  # [None, feat_dim]
    scatter_idxs = tf.boolean_mask(scatter_vals, mask)  # [None, 2]
    # # Averages any cases where tokens are in multiple cells - for example when cells span multiple rows / cols.
    divide_by = tf.tensor_scatter_nd_add(
        input_tensor, scatter_idxs, tf.ones_like(feats)
    )  # [text_batch, text_seq, feat_dim]
    return tf.math.divide_no_nan(
        tf.tensor_scatter_nd_add(input_tensor, scatter_idxs, feats), divide_by
    )  # [text_batch, text_seq, feat_dim]


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


class TableModelBatchPostprocessor:
    def __init__(self, config=None):
        self.config = config
        self._cached_stats = None  # (row_len, col_len, row_bs, col_bs)

    def modify_input_spec(self, input_spec):
        (types, target_type), (shapes, target_shape) = input_spec
        gather_shapes = {
            "seq_lens": tf.TensorShape([None]),
            "values": tf.TensorShape([None, None, 2]),
            "attn_mask": tf.TensorShape([None, None, None]),
            "pos_ids": tf.TensorShape([None, None]),
        }
        gather_types = {
            "seq_lens": tf.int32,
            "values": tf.int32,
            "attn_mask": tf.float32,
            "pos_ids": tf.int32,
        }
        shapes["row_gather"] = gather_shapes
        shapes["col_gather"] = gather_shapes
        types["row_gather"] = gather_types
        types["col_gather"] = gather_types
        return (types, target_type), (shapes, target_shape)

    def _pad_predict_batch(self, feature_batch):
        batch_size = len(feature_batch)
        max_seq_len = max(len(features["tokens"]) for features in feature_batch)
        tokens = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        context = np.zeros(
            (batch_size, max_seq_len, self.config.context_dim), dtype=np.float32
        )
        lengths = np.zeros((batch_size,), dtype=np.int32)

        for idx, features in enumerate(feature_batch):
            seq_len = len(features["tokens"])
            tokens[idx, :seq_len] = features["tokens"]
            context[idx, :seq_len] = features["context"]
            lengths[idx] = seq_len

        return {
            "tokens": tf.convert_to_tensor(tokens, dtype=tf.int32),
            "context": tf.convert_to_tensor(context, dtype=tf.float32),
            "length": tf.convert_to_tensor(lengths, dtype=tf.int32),
        }

    def iter_predict_batches(self, feature_iter, predict_batch_size):
        batch = []
        for features in feature_iter:
            batch.append(features)
            if len(batch) == predict_batch_size:
                yield self._postprocess_inference(self._pad_predict_batch(batch))
                batch = []
        if batch:
            yield self._postprocess_inference(self._pad_predict_batch(batch))

    def _postprocess_inference(self, x):
        with tf.device("/CPU:0"):
            end_col, end_row, start_col, start_row = tf.unstack(
                tf.cast(x["context"], tf.int32), num=4, axis=2
            )
            row_gather = _build_gather_outputs_inference_numpy(
                X=x["tokens"],
                sequence_lengths=x["length"],
                start=start_row,
                end=end_row,
                other_end=end_col,
                chunk_tables=self.config.chunk_tables,
                include_mask=True,
            )
            col_gather = _build_gather_outputs_inference_numpy(
                X=x["tokens"],
                sequence_lengths=x["length"],
                start=start_col,
                end=end_col,
                other_end=end_row,
                chunk_tables=self.config.chunk_tables,
                include_mask=True,
            )
            return {**x, "row_gather": row_gather, "col_gather": col_gather}

    def _postprocess(
        self,
        x,
        y=None,
        target_row_len=None,
        target_col_len=None,
        target_row_batch_size=None,
        target_col_batch_size=None,
        training=True,
    ):
        # Should run on cpu anyway but just to be safe.
        with tf.device("/CPU:0"):
            end_col, end_row, start_col, start_row = tf.unstack(
                tf.cast(x["context"], tf.int32), num=4, axis=2
            )
            # Get gather indices for rows and columns
            row_gather = build_gather_outputs(
                X=x["tokens"],
                sequence_lengths=x["length"],
                start=start_row,
                end=end_row,
                other_end=end_col,
                chunk_tables=self.config.chunk_tables,
                include_mask=True,
                training=training,
                target_seq_len=target_row_len,
                target_batch_size=target_row_batch_size,
            )
            col_gather = build_gather_outputs(
                X=x["tokens"],
                sequence_lengths=x["length"],
                start=start_col,
                end=end_col,
                other_end=end_row,
                chunk_tables=self.config.chunk_tables,
                include_mask=True,
                training=training,
                target_seq_len=target_col_len,
                target_batch_size=target_col_batch_size,
            )
            x = {**x, "row_gather": row_gather, "col_gather": col_gather}
            if y is not None:
                return x, y
            return x

    # Backward-compatible default mapping fn (assumes training semantics)
    def postprocess(self, x, y=None):
        return self._postprocess(x, y=y, training=True)

    def get_dataset_transform(self, mode="train"):
        assert mode in {"train", "predict"}

        def _transform(ds):
            # For training, scan dataset to set global target lengths/batch sizes
            if mode == "train":
                # If we already computed stats for this postprocessor, just reuse them
                if self._cached_stats is None:
                    LOGGER.info(
                        "Running down 1 epoch of the dataset to calculate optimal batch size and sequence length for table model."
                    )

                    def _stats_map(*args):
                        if len(args) == 2:
                            x, _y = args
                        else:
                            (x,) = args
                        end_col, end_row, start_col, start_row = tf.unstack(
                            tf.cast(x["context"], tf.int32), num=4, axis=2
                        )
                        ragged_rows = compute_masked_ragged_indices(
                            x["tokens"],
                            x["length"],
                            start_row,
                            end_row,
                            other_end=end_col,
                            chunk_tables=self.config.chunk_tables,
                        )
                        ragged_cols = compute_masked_ragged_indices(
                            x["tokens"],
                            x["length"],
                            start_col,
                            end_col,
                            other_end=end_row,
                            chunk_tables=self.config.chunk_tables,
                        )
                        return ragged_rows, ragged_cols

                    ragged_values = ds.map(
                        _stats_map,
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False,
                    )

                    # Compute maxima across the bounded window
                    max_row_seq_len = 0
                    max_col_seq_len = 0
                    for ragged_rows, ragged_cols in ragged_values:
                        max_row_seq_len = max(
                            max_row_seq_len,
                            int(tf.reduce_max(ragged_rows.row_lengths()).numpy()),
                        )
                        max_col_seq_len = max(
                            max_col_seq_len,
                            int(tf.reduce_max(ragged_cols.row_lengths()).numpy()),
                        )

                    # Compute packed batch sizes using those maxima
                    def _batch_packing_map(ragged_rows, ragged_cols):
                        repacked_rows, _, _ = batch_packing(
                            ragged_rows,
                            include_mask=False,
                            training=True,
                            target_seq_len=max_row_seq_len,
                        )
                        repacked_cols, _, _ = batch_packing(
                            ragged_cols,
                            include_mask=False,
                            training=True,
                            target_seq_len=max_col_seq_len,
                        )
                        return tf.shape(repacked_rows)[0], tf.shape(repacked_cols)[0]

                    row_batch_size = 0
                    col_batch_size = 0
                    for row_bs, col_bs in ragged_values.map(
                        _batch_packing_map, num_parallel_calls=tf.data.AUTOTUNE
                    ):
                        row_batch_size = max(row_batch_size, int(row_bs.numpy()))
                        col_batch_size = max(col_batch_size, int(col_bs.numpy()))

                    self._cached_stats = (
                        int(max_row_seq_len),
                        int(max_col_seq_len),
                        int(row_batch_size),
                        int(col_batch_size),
                    )
                    LOGGER.info(
                        f"Completed calculating stats. Cached stats: {self._cached_stats}"
                    )
                else:
                    LOGGER.info("Reusing cached stats")

                # Unpack cached stats
                (
                    max_row_seq_len,
                    max_col_seq_len,
                    row_batch_size,
                    col_batch_size,
                ) = self._cached_stats
                target_col_batch_size = tf.convert_to_tensor(
                    col_batch_size, dtype=tf.int32
                )
                target_row_batch_size = tf.convert_to_tensor(
                    row_batch_size, dtype=tf.int32
                )
                target_col_len = tf.convert_to_tensor(max_col_seq_len, dtype=tf.int32)
                target_row_len = tf.convert_to_tensor(max_row_seq_len, dtype=tf.int32)

                def _train_map(*args):
                    if len(args) == 2:
                        x, y = args
                    else:
                        (x,) = args
                        y = None
                    return self._postprocess(
                        x,
                        y=y,
                        target_row_len=target_row_len,
                        target_col_len=target_col_len,
                        target_row_batch_size=target_row_batch_size,
                        target_col_batch_size=target_col_batch_size,
                        training=True,
                    )

                return ds.map(_train_map, num_parallel_calls=tf.data.AUTOTUNE)

            # Predict/test mode: use dynamic packing; no pre-scan
            return ds.map(
                lambda *args: self._postprocess(*args, training=False),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return _transform

import functools
import math

import tensorflow as tf
from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder
from finetune.base_models.bert.modeling import BertConfig, BertModel, LayoutLMModel, \
    create_initializer, get_shape_list, reshape_to_matrix


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

    eos_idx = tf.argmax(
        input=tf.cast(delimiters, tf.float32)
        * tf.expand_dims(
            tf.range(tf.cast(seq_length, tf.float32), dtype=tf.float32), 0
        ),
        axis=1,
    )

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
            bert.get_sequence_output(),
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


def long_doc_bert_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    context=None,
    total_num_steps=None,
    underlying_model=BertModel,
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
        features: The output of the featurizer_final state.
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
        # FIXME Running into errors with max_position_embeddings arg
        # max_position_embeddings=config.max_length,
        # max_position_embeddings=config.chunk_size,
        type_vocab_size=2,
        initializer_range=config.weight_stddev,
        low_memory_mode=config.low_memory_mode,
        pos_injection=config.context_injection,
        reading_order_removed=config.reading_order_removed,
        anneal_reading_order=config.anneal_reading_order,
        positional_channels=config.context_channels,
    )

    initial_shape = tf.shape(input=X)

    # [batch_size, sequence_length]
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))
    X.set_shape([None, None])
    if config.extra_print:
        X = tf.compat.v1.Print(X, ["X", X, tf.shape(X)], summarize=-1)

    # Remove start tokens from tensors, because they will be added back per chunk
    batch_size = initial_shape[0]
    sequence_len = initial_shape[-1] - 1
    X = X[:, 1:]
    if config.extra_print:
        X = tf.compat.v1.Print(X, ["X_remove_start", X, tf.shape(X)], summarize=-1)

    # Add padding to X to later break out a separate chunk dimension
    """
    FIXME Remove comments with math. Those are with these assumptions for validating
    padding / chunking logic
    * config.chunk_size = 64
    * config.batch_size_scaler = 4
    * sequence_len = 300 - 1 = 299
    """
    # Need to effectively "scale" chunk dim to account for division by batch_size_scaler so that
    # num_chunks is divisible by batch_size_scaler. In some cases this requires padding by a lot
    # of zeros
    effective_chunk_dim = config.chunk_size * config.batch_size_scaler    # 64 * 4 = 256
    pad_count = effective_chunk_dim - (sequence_len % effective_chunk_dim)  # 256 - (299 % 256) = 197
    new_seq_len = sequence_len + pad_count  # 299 + 213 = 512
    num_chunks = tf.cast(new_seq_len / config.chunk_size, dtype=tf.int32)  # 512 / 64 = 8
    zero_paddings = tf.convert_to_tensor([[0, 0], [0, pad_count]], dtype=X.dtype)
    X = tf.pad(X, zero_paddings)
    if config.extra_print:
        X = tf.compat.v1.Print(X, ["X_pad", X, tf.shape(X)], summarize=-1)

    if config.num_layers_trained not in [config.n_layer, 0]:
        raise ValueError(
            "Bert base model does not support num_layers_trained not equal to 0 or n_layer"
        )

    if config.anneal_reading_order:
        reading_order_decay_rate = get_decay_for_half(total_num_steps)
    else:
        reading_order_decay_rate = None

    # Reshape to [batch_size, num_chunks, chunk_size]
    X = tf.reshape(X, [batch_size, num_chunks, config.chunk_size])

    # Reshape everything to
    # [batch_size * batch_scaling, num_chunks / batch_scaling, chunk_size]
    # Purpose is to parallelize as much as we can through each pass to bert_wrapper function
    # during map_fn, without running OOM
    X = tf.reshape(
        X,
        [
            batch_size * config.batch_size_scaler,
            tf.cast(num_chunks / config.batch_size_scaler, dtype=tf.int32),
            config.chunk_size,
        ],
    )
    if config.extra_print:
        X = tf.compat.v1.Print(X, ["X_batch_scaler", X, tf.shape(X)], summarize=-1)

    # Add start/end tokens to each chunk for BERT featurizer. This gets the shape to
    # [batch_size * batch_scaling, num_chunks / batch_scaling, new_chunk_size],
    # where new_chunk_size = chunk_size + 2
    # This means that the effective sequence length is now num_chunks new_chunk_size
    effective_seq_len = num_chunks * (config.chunk_size + 2)
    start_token_pad = tf.constant([[0, 0], [0, 0], [1, 0]])
    end_token_pad = tf.constant([[0, 0], [0, 0], [0, 1]])
    X = tf.pad(X, start_token_pad, constant_values=encoder.start_token)

    # Need to create pad mask across effective sequence length now that we have multiple
    # end tokens, one for each chunk. Want to create a mask based on the last one.
    # The first padding operation is a hack to get the shape correct without adding
    # end tokens to each chunk, which make it impossible to determine the true end of
    # the sequence when there are chunk(s) at the end that are all padding.
    X_pad = tf.pad(X, end_token_pad)
    # First need to reshape X to [batch_size, effective_seq_len)
    X_pad = tf.reshape(X_pad, [batch_size, effective_seq_len])
    delimiters = tf.cast(tf.equal(X_pad, encoder.delimiter_token), tf.int32)
    seq_length = tf.shape(input=delimiters)[1]
    eos_idx = tf.argmax(
        input=tf.cast(delimiters, tf.float32)
              * tf.expand_dims(
            tf.range(tf.cast(seq_length, tf.float32), dtype=tf.float32), 0
        ),
        axis=1,
    )
    lengths = lengths_from_eos_idx(eos_idx=eos_idx, max_length=seq_length)
    pad_mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)
    # Reshape mask to [batch_size, effective_seq_len, 1]
    pad_mask = tf.expand_dims(pad_mask, axis=-1)

    # Need to calculate pad_mask first, because the end_token and delimiter_token are the same
    X = tf.pad(X, end_token_pad, constant_values=encoder.end_token)
    if config.extra_print:
        X = tf.compat.v1.Print(X, ["X_start_end_pad", X, tf.shape(X),
                                   "pad_mask", tf.shape(pad_mask), pad_mask], summarize=-1)

    # Then transpose so that the chunk dim comes first since map_fn iterates over first dim
    # [num_chunks / batch_scaling, batch_size * batch_scaling, new_chunk_size]
    X = tf.transpose(X, [1, 0, 2])
    if config.extra_print:
        X = tf.compat.v1.Print(X, ["X_transpose", X, tf.shape(X)], summarize=-1)

    def bert_wrapper(X_sub):
        """
        Target for map_fn to get BERT embeddings for each chunk in a long
        sequence. X_sub, etc are sliced along the chunk dim

        X_sub [batch_size * batch_scaling, new_chunk_size]

        Return sequence features and mask from BERT
        """
        if config.extra_print:
            X_sub = tf.compat.v1.Print(
                X_sub, ["X_sub", X_sub, tf.shape(X_sub)], summarize=-1
            )
        # To fit the interface of finetune we are going to compute the mask and type id at runtime.
        delimiters = tf.cast(tf.equal(X_sub, encoder.delimiter_token), dtype=tf.int32)
        if config.extra_print:
            delimiters = tf.compat.v1.Print(
                delimiters, ["delimiters", delimiters, tf.shape(delimiters)], summarize=-1
            )
        # Appears to be 0 where you have tokens and 1 where you have padding
        # [batch_size * batch_scaling, new_chunk_size]
        token_type_ids = tf.cumsum(delimiters, exclusive=True, axis=1)
        if config.extra_print:
            token_type_ids = tf.compat.v1.Print(
                token_type_ids, ["token_type_ids", token_type_ids, tf.shape(token_type_ids)], summarize=-1
            )

        seq_length = tf.shape(input=delimiters)[1]
        # We explicitly add start and end tokens to each chunk, so in some cases we will
        # have duplicate end/delimiter tokens in a chunk. In that case, we want to select
        # the first instance, because that actually indicates where the sequence ends and
        # where padding starts. Note that this approach is problematic for multi input tasks
        # tf.argmax chooses the smallest index in the case of ties
        eos_idx = tf.argmax(input=delimiters, axis=1)
        if config.extra_print:
            eos_idx = tf.compat.v1.Print(eos_idx, ["eos_idx", eos_idx, tf.shape(eos_idx)], summarize=-1)

        lengths = lengths_from_eos_idx(eos_idx=eos_idx, max_length=seq_length)
        if config.extra_print:
            lengths = tf.compat.v1.Print(lengths, ["lengths", lengths, tf.shape(lengths),
                                                   "seq_length", seq_length], summarize=-1)

        # Appears to be 1 where you have tokens and 0 where you have padding
        # [batch_size * batch_scaling, new_chunk_size]
        mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)
        if config.extra_print:
            mask = tf.compat.v1.Print(mask, ["mask", mask, tf.shape(mask)], summarize=-1)

        bert = underlying_model(
            config=bert_config,
            is_training=train,
            input_ids=X_sub,
            input_mask=mask,
            input_context=context,
            token_type_ids=token_type_ids,
            use_one_hot_embeddings=False,
            scope=None,
            use_pooler=config.bert_use_pooler,
            use_token_type=config.bert_use_type_embed,
            reading_order_decay_rate=reading_order_decay_rate,
        )
        sub_seq = bert.get_sequence_output()
        if config.extra_print:
            sub_seq = tf.compat.v1.Print(
                sub_seq, ["sub_seq", tf.shape(sub_seq)], summarize=-1
            )
        return sub_seq

    def chunk_aggregation(seq_output, seq_mask):
        """
        Function for aggregating/pooling the pooled outputs from the BERT featurizer
        across the chunk dimension

        feature_size = hidden_size for mean, max, attention
        feature_size = hidden_size * 2 for concat

        Args:
            seq_output: [batch_size, effective_seq_len, hidden_size]
            seq_mask: [batch_size, effective_seq_len, 1] - Mask is 1 where
                you have tokens, and 0 otherwise

        Returns:
            aggr_outputs: [batch_size, feature_size]

        TODO Implement LSTM
        """
        seq_mask = tf.cast(seq_mask, dtype=seq_output.dtype)
        # Invert mask so that it is 0 where you have tokens, and 1 otherwise
        inv_seq_mask = 1. - seq_mask
        # Create separate projections for mean/max
        if config.conv_proj:
            seq_proj_mean = tf.compat.v1.layers.conv1d(
                inputs=seq_output,
                filters=config.n_embed,
                # width
                kernel_size=3,
                # so output tensor has the same shape
                padding="same",
                kernel_initializer=create_initializer(config.weight_stddev),
                name="mean_proj_conv_not_chunk_attn"
            )
            seq_proj_max = tf.compat.v1.layers.conv1d(
                inputs=seq_output,
                filters=config.n_embed,
                kernel_size=3,
                padding="same",
                kernel_initializer=create_initializer(config.weight_stddev),
                name="max_proj_conv_not_chunk_attn"
            )
        else:
            seq_proj_mean = tf.compat.v1.layers.dense(
                inputs=seq_output,
                units=config.n_embed,
                kernel_initializer=create_initializer(config.weight_stddev),
                name="mean_proj_not_chunk_attn"
            )
            seq_proj_max = tf.compat.v1.layers.dense(
                inputs=seq_output,
                units=config.n_embed,
                kernel_initializer=create_initializer(config.weight_stddev),
                name="max_proj_not_chunk_attn"
            )

        if config.chunk_pool_fn == "mean":
            denom = tf.reduce_sum(seq_mask, axis=1)
            aggr_outputs = tf.reduce_sum(seq_proj_mean * seq_mask, axis=1) / denom
        elif config.chunk_pool_fn == "attention":
            # aggr_outputs = attn(seq_output, config, seq_mask)
            aggr_outputs = attention_layer(
                to_tensor=seq_output,
                num_attention_heads=config.aggr_attn_heads,
                size_per_head=config.n_embed,
                attention_mask=seq_mask,
                do_return_2d_tensor=True,
                extra_print=config.extra_print,
            )
        elif config.chunk_pool_fn == "max":
            # Make padded values very large negative numbers
            aggr_outputs = tf.reduce_max(seq_proj_max - (inv_seq_mask * 1e9), axis=1)
        elif config.chunk_pool_fn in {"concat", "concat_attn"}:
            mean_denom = tf.reduce_sum(seq_mask, axis=1)
            aggr_mean = tf.reduce_sum(seq_proj_mean * seq_mask, axis=1) / mean_denom
            aggr_max = tf.reduce_max(seq_proj_max - (inv_seq_mask * 1e9), axis=1)
            if config.chunk_pool_fn == "concat_attn":
                # aggr_attn = attn(seq_output, config, seq_mask)
                aggr_attn = attention_layer(
                    to_tensor=seq_output,
                    num_attention_heads=config.aggr_attn_heads,
                    size_per_head=config.n_embed,
                    attention_mask=seq_mask,
                    do_return_2d_tensor=True,
                    extra_print=config.extra_print,
                )
                aggr_outputs = tf.concat([aggr_mean, aggr_max, aggr_attn], axis=1)
            else:
                # Concat across the hidden_size dim (axis=1) so feature_size dim is hidden_size * 2
                aggr_outputs = tf.concat([aggr_mean, aggr_max], axis=1)
        else:
            raise ValueError(f"chunk_pool_fn={config.chunk_pool_fn} is not supported")

        return aggr_outputs

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        """
        * Get sequence features out of BERT per chunk [chunk_size, hidden_size]
        * Reshape to [batch_size, sequence_len, hidden_size], because seq_len = chunk_size * num_chunks
        * Aggregate across full sequence length
        """
        # Send each chunk through map_fn that outputs BERT sequence embeddings (seq_output)
        # [num_chunks / batch_scaling, batch_size * batch_scaling, chunk_size, hidden_size]
        seq_output = tf.map_fn(bert_wrapper, X, fn_output_signature=tf.float32)

        # We want static embeddings from BERT, so not computing gradients
        seq_output = tf.stop_gradient(seq_output)

        if config.extra_print:
            seq_output = tf.compat.v1.Print(
                seq_output, ["seq_output", tf.shape(seq_output)], summarize=-1
            )

        # Transpose to [batch_size * batch_scaling, num_chunks / batch_scaling, chunk_size, hidden_size]
        seq_output = tf.transpose(seq_output, [1, 0, 2, 3])

        # Reshape output to remove batch_scaling factor and combine num_chunks and
        # chunk_size into sequence length
        # Note that effective_seq_len is longer than original sequence length due to
        # the start/end tokens added to each chunk prior to featurization
        # [batch_size, effective_seq_len, hidden_size]
        seq_output = tf.reshape(
            seq_output,
            [
                batch_size,
                effective_seq_len,
                config.n_embed,
            ],
        )
        if config.extra_print:
            seq_output = tf.compat.v1.Print(
                seq_output, ["seq_rs", tf.shape(seq_output)], summarize=-1
            )

        # Learn a projection on the BERT embeddings prior to aggregation. Not necessary
        # if the chunk_pool_fn is attention, because that already has learnable projections
        # # [batch_size, num_chunks, hidden_size]
        # [batch_size, seq_len, hidden_size]
        # if config.chunk_pool_fn != "attention":
        #     proj_output = tf.compat.v1.layers.dense(
        #         inputs=seq_output,
        #         units=config.n_embed,
        #         kernel_initializer=create_initializer(config.weight_stddev),
        #         # FIXME This name only has chunk_attn is because I'm too lazy to fix regex
        #         # on permit_uninitialized in base model
        #         name="proj_not_chunk_attn"
        #     )
        # else:
        #     proj_output = seq_output

        # Reduce across chunk dim with aggregation operation [batch_size, feature_size]
        # features = chunk_aggregation(proj_output, seq_mask)
        # TODO Could try dropout prior to max pooling operation. Drop out specific tokens
        # Use dropout mask mask_shape = [batch, seq, 1] instead of [..., seq, feats]
        features = chunk_aggregation(seq_output, pad_mask)
        if config.extra_print:
            features = tf.compat.v1.Print(
                features, ["features", tf.shape(features), features[:, :4]], summarize=-1
            )

        # Ensure that batch dim(s) are properly shaped. Required for target models
        # with multiple batch dims
        features = tf.reshape(
            features,
            shape=tf.concat((initial_shape[:-1], [tf.shape(features)[-1]]), 0),
        )

        output_state = {
            "features": features,
            # "embed_weights": embed_weights
            # "lengths": lengths,
            # "eos_idx": eos_idx,
            # Sequence features required in base model_fn, but unused for classifier
            # TODO Do I need to add sequence features for any other target models?
            # Setting to features so I don't get type errors for None
            "sequence_features": features,
        }
        if config.num_layers_trained == 0:
            output_state = {k: tf.stop_gradient(v) for k, v in output_state.items()}

        return output_state


def attn(hidden, config, seq_mask):
    """
    Args:
        hidden: [batch_size, effective_seq_len, feature_size]
        seq_mask: [batch_size, effective_seq_len, 1] - 1 for tokens, 0 for padding
    """
    # Shapes
    # FIXME Running into shape errors trying to use this, so using config.n_embed instead
    # TypeError: Dimension value must be integer or None or have an __index__ method, got value '<tf.Tensor...
    # hidden_shape = get_shape_list(hidden)
    # hidden_size = hidden_shape[-1]

    # Define learnable key and value projections
    # Don't need batch size because same parameters are used for every
    # element in the batch. Typically only see batch sizes in activations

    # Typically key_proj is hidden_size x hidden_size, and output
    # is divided into the individual heads
    key_proj = tf.compat.v1.get_variable(
        name="chunk_attn_key_proj",
        # In future, I could try [hidden_size, hidden_size * num_attn_heads]
        shape=[config.n_embed, config.n_embed],
        dtype=tf.float32,
        trainable=True,
        initializer=create_initializer(config.weight_stddev)
    )
    value_proj = tf.compat.v1.get_variable(
        name="chunk_attn_value_proj",
        shape=[config.n_embed, config.n_embed],
        dtype=tf.float32,
        trainable=True,
        initializer=create_initializer(config.weight_stddev)
    )

    # Define learnable query vector
    query = tf.compat.v1.get_variable(
        name="chunk_attn_query",
        shape=[1, config.n_embed],
        dtype=tf.float32,
        trainable=True,
        initializer=create_initializer(config.weight_stddev)
    )

    # Compute keys and values
    # For mat muls just thing about the last two axes. Everything else should be the same
    # [batch_size, n_chunks, hidden_size] * [hidden_size x hidden_size]
    # = [batch_size x n_chunks x hidden_size]
    keys = tf.matmul(hidden, key_proj)
    values = tf.matmul(hidden, value_proj)

    # Reshape last dimension of keys hidden_size into number of heads * head size
    # [batch_size, n_chunks, num_heads, head_size] for both keys/values
    # Then reshape to [batch_size, num_heads, n_chunks, head_size]

    # Learn query to be [head_size, num_heads, 1] for attn_matrix mat/mul

    # Compute attention matrix
    # [batch_size x n_chunks x hidden_size] * [hidden_size, 1] (transposed)
    # = [batch_size x n_chunks x 1]
    attn_matrix = tf.matmul(keys, query, transpose_b=True)

    # Make attention a very large negative value where we have a pad token
    attn_matrix = attn_matrix - ((1 - seq_mask) * 1e9)

    # Take softmax
    attn_matrix = tf.nn.softmax(attn_matrix, axis=1)

    # Multiply values by attn_matrix to get output representation
    # [batch_size x hidden_size x n_chunks] (transposed) * [batch_size x n_chunks x 1]
    # = [batch_size x hidden_size x 1]
    output = tf.matmul(values, attn_matrix, transpose_a=True)

    # Squeeze output to [batch_size x hidden_size]
    output = tf.squeeze(output, axis=2)
    return output


def attention_layer(
        to_tensor,
        attention_mask=None,
        num_attention_heads=4,
        size_per_head=768,
        key_act=None,
        value_act=None,
        initializer_range=0.02,
        do_return_2d_tensor=False,
        extra_print=False,
):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size, from_seq_length, 1].
            The values should be 1 or 0. The attention scores will effectively be set
            to -infinity for any positions in the mask that are 0, and will be unchanged for
            positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].

    Returns:
        float Tensor of shape [batch_size, 1, num_attention_heads * size_per_head].
            (If `do_return_2d_tensor` is true, this will be of shape
            [batch_size, num_attention_heads * size_per_head]).
    """

    def transpose_for_scores(
            input_tensor, batch_size, num_attention_heads, seq_length, width
    ):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width]
        )

        output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
        return output_tensor

    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    batch_size = to_shape[0]
    to_seq_length = to_shape[1]

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length - hard coding this to 1 so that
    #       we have one query per head rather than per token
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [1, N*H]
    query_layer = tf.compat.v1.get_variable(
        name="chunk_attn_query_multihead",
        shape=[1, num_attention_heads * size_per_head],
        dtype=tf.float32,
        trainable=True,
        initializer=create_initializer(initializer_range)
    )
    if extra_print:
        query_layer = tf.compat.v1.Print(
            query_layer, ["query_layer", tf.shape(query_layer)], summarize=-1
        )

    # `key_layer` = [B*T, N*H]
    key_layer = tf.compat.v1.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key_not_chunk_attn",
        kernel_initializer=create_initializer(initializer_range),
    )
    if extra_print:
        key_layer = tf.compat.v1.Print(
            key_layer, ["key_layer", tf.shape(key_layer)], summarize=-1
        )

    # `value_layer` = [B*T, N*H]
    value_layer = tf.compat.v1.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value_not_chunk_attn",
        kernel_initializer=create_initializer(initializer_range),
    )
    if extra_print:
        value_layer = tf.compat.v1.Print(
            value_layer, ["value_layer", tf.shape(value_layer)], summarize=-1
        )

    # `query_layer` = [1, N, 1, H]
    query_layer = transpose_for_scores(
        query_layer, 1, num_attention_heads, 1, size_per_head
    )
    if extra_print:
        query_layer = tf.compat.v1.Print(
            query_layer, ["query_layer_tp", tf.shape(query_layer)], summarize=-1
        )

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(
        key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head
    )
    if extra_print:
        key_layer = tf.compat.v1.Print(
            key_layer, ["key_layer_tp", tf.shape(key_layer)], summarize=-1
        )

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # [1, N, 1, H] * [B, N, H, T] = [B, N, 1, T]
    # `attention_scores` = [B, N, 1, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(
        attention_scores, 1.0 / math.sqrt(float(size_per_head))
    )
    if extra_print:
        attention_scores = tf.compat.v1.Print(
            attention_scores, ["attention_scores", tf.shape(attention_scores)], summarize=-1
        )

    if attention_mask is not None:
        # Starting dims are [B, T, 1]. Need to reshape to [B, 1, 1, T]
        # After expanding, dims `attention_mask` = [B, T, 1, 1]
        attention_mask = tf.expand_dims(attention_mask, axis=-1)
        # After reshape, [B, 1, 1, T]
        attention_mask = tf.transpose(a=attention_mask, perm=[0, 2, 3, 1])
        if extra_print:
            attention_mask = tf.compat.v1.Print(
                attention_mask, ["attention_mask", tf.shape(attention_mask)], summarize=-1
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, 1, T]
    attention_probs = tf.nn.softmax(attention_scores)
    if extra_print:
        attention_probs = tf.compat.v1.Print(
            attention_probs, ["attention_probs", tf.shape(attention_probs)], summarize=-1
        )

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head]
    )
    if extra_print:
        value_layer = tf.compat.v1.Print(
            value_layer, ["value_layer_rshp", tf.shape(value_layer)], summarize=-1
        )

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])
    if extra_print:
        value_layer = tf.compat.v1.Print(
            value_layer, ["value_layer_tp", tf.shape(value_layer)], summarize=-1
        )

    # [B, N, 1, T] * [B, N, T, H] = [B, N, 1, H]
    # `context_layer` = [B, N, 1, H]
    context_layer = tf.matmul(attention_probs, value_layer)
    if extra_print:
        context_layer = tf.compat.v1.Print(
            context_layer, ["context_layer", tf.shape(context_layer)], summarize=-1
        )

    # `context_layer` = [B, 1, N, H]
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
    if extra_print:
        context_layer = tf.compat.v1.Print(
            context_layer, ["context_layer_tp", tf.shape(context_layer)], summarize=-1
        )

    if do_return_2d_tensor:
        # `context_layer` = [B*1, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, num_attention_heads * size_per_head],
        )
    else:
        # `context_layer` = [B, 1, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, 1, num_attention_heads * size_per_head],
        )

    return context_layer


layoutlm_featurizer = functools.partial(bert_featurizer, underlying_model=LayoutLMModel)

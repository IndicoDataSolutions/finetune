import functools
import math

import tensorflow as tf
from finetune.util.shapes import lengths_from_eos_idx
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder
from finetune.base_models.bert.modeling import BertConfig, BertModel, LayoutLMModel


def get_decay_for_half(total_num_steps):
    decay = tf.minimum(tf.cast(tf.compat.v1.train.get_global_step(), tf.float32) / (total_num_steps / 2), 1.0)
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
    model_filename = config.base_model_path.rpartition('/')[-1]
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
        batch_scaling=2,
        train=False,
        reuse=None,
        context=None,
        total_num_steps=None,
        underlying_model=BertModel,
        **kwargs
):
    """
    The transformer element of the finetuning model. Maps from tokens ids to a dense, embedding of the sequence.

    # TODO Do I need to do anything to get static BERT embeddings and then only train the pooling operation?

    :param X: A tensor of token indexes with shape [batch_size, sequence_length, token_idx]
    :param encoder: A TextEncoder object.
    :param config: A config object, containing all parameters for the featurizer.
    :param batch_scaling: Factor by which to scale up batch dimension during map_fn for better parallelism
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
        low_memory_mode=config.low_memory_mode,
        pos_injection=config.context_injection,
        reading_order_removed=config.reading_order_removed,
        anneal_reading_order=config.anneal_reading_order,
        positional_channels=config.context_channels,
    )

    initial_shape = tf.shape(input=X)
    initial_shape = tf.compat.v1.Print(initial_shape, ["initial_shape", initial_shape], summarize=-1)
    batch_size = initial_shape[0]
    batch_size = tf.compat.v1.Print(batch_size, ["batch_size", batch_size], summarize=-1)
    sequence_len = initial_shape[1]
    sequence_len = tf.compat.v1.Print(sequence_len, ["sequence_len", sequence_len], summarize=-1)

    X = tf.compat.v1.Print(X, ["X_init", X, tf.shape(X)], summarize=-1)
    # [batch_size, sequence_length]
    X = tf.reshape(X, shape=tf.concat(([-1], initial_shape[-1:]), 0))
    X.set_shape([None, None])
    X = tf.compat.v1.Print(X, ["X", X, tf.shape(X)], summarize=-1)

    # Add padding to X to later break out a separate chunk dimension
    pad_count = config.max_chunk_length - (sequence_len % config.max_chunk_length)
    new_seq_len = sequence_len + pad_count
    num_chunks = new_seq_len / config.max_chunk_length
    zero_pad = tf.zeros([batch_size, pad_count], dtype=X.dtype)
    X = tf.concat([X, zero_pad], axis=1)
    X = tf.compat.v1.Print(X, ["X_pad", X, tf.shape(X)], summarize=-1)

    # # To fit the interface of finetune we are going to compute the mask and type id at runtime.
    # delimiters = tf.cast(tf.equal(X, encoder.delimiter_token), tf.int32)
    #
    # # Appears to be 0 where you have tokens and 1 where you have padding
    # # [batch_size, sequence_length]
    # token_type_ids = tf.cumsum(delimiters, exclusive=True, axis=1)
    #
    # seq_length = tf.shape(input=delimiters)[1]
    #
    # eos_idx = tf.argmax(
    #     input=tf.cast(delimiters, tf.float32)
    #           * tf.expand_dims(
    #         tf.range(tf.cast(seq_length, tf.float32), dtype=tf.float32), 0
    #     ),
    #     axis=1,
    # )
    # eos_idx = tf.compat.v1.Print(eos_idx, ["eos_idx", eos_idx, tf.shape(eos_idx)], summarize=-1)
    #
    # lengths = lengths_from_eos_idx(eos_idx=eos_idx, max_length=seq_length)
    # lengths = tf.compat.v1.Print(lengths, ["lengths", lengths, tf.shape(lengths)], summarize=-1)
    #
    # # Appears to be 1 where you have tokens and 0 where you have padding
    # # [batch_size, sequence_length]
    # mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)

    if config.num_layers_trained not in [config.n_layer, 0]:
        raise ValueError(
            "Bert base model does not support num_layers_trained not equal to 0 or n_layer"
        )

    if config.anneal_reading_order:
        reading_order_decay_rate = get_decay_for_half(total_num_steps)
    else:
        reading_order_decay_rate = None

    # X = tf.compat.v1.Print(X, ["X", X, tf.shape(X)], summarize=-1)
    # mask = tf.compat.v1.Print(mask, ["mask", mask, tf.shape(mask)], summarize=-1)
    # token_type_ids = tf.compat.v1.Print(token_type_ids, ["token_type_ids", token_type_ids, tf.shape(token_type_ids)], summarize=-1)

    # Reshape everything to
    # [batch_size * batch_scaling, num_chunks / batch_scaling, max_chunk_len]
    # Purpose is to parallelize as much as we can through each pass to bert_wrapper function
    # during map_fn, without running OOM
    # TODO This math breaks down if num_chunks is odd
    # Fixing that might require more sophisticating padding to make sure num_chunks is always even
    X = tf.reshape(X, [batch_size*batch_scaling,
                       tf.cast(num_chunks/batch_scaling, dtype=tf.int32), config.max_chunk_length])
    X = tf.compat.v1.Print(X, ["X_bscale", X, tf.shape(X)], summarize=-1)
    # mask = tf.reshape(mask, [batch_size * batch_scaling,
    #                          tf.cast(num_chunks/batch_scaling, dtype=tf.int32), config.max_chunk_length])
    # token_type_ids = tf.reshape(token_type_ids,
    #                             [batch_size * batch_scaling,
    #                              tf.cast(num_chunks/batch_scaling, dtype=tf.int32), config.max_chunk_length])

    # Then transpose so that the chunk dim comes first since map_fn iterates over first dim
    # [num_chunks / batch_scaling, batch_size * batch_scaling, max_chunk_len]
    X = tf.transpose(X, [1, 0, 2])
    # mask = tf.transpose(mask, [1, 0, 2])
    # token_type_ids = tf.transpose(token_type_ids, [1, 0, 2])

    X = tf.compat.v1.Print(X, ["X_tp", X, tf.shape(X)], summarize=-1)
    # mask = tf.compat.v1.Print(mask, ["mask_tp", mask, tf.shape(mask)], summarize=-1)
    # token_type_ids = tf.compat.v1.Print(token_type_ids, ["token_type_ids_tp", token_type_ids, tf.shape(token_type_ids)],
    #                                     summarize=-1)

    # def bert_wrapper(X_sub, mask_sub, token_type_ids_sub):
    def bert_wrapper(X_sub):
        """
        Target for map_fn to get BERT embeddings for each chunk in a long
        sequence. X_sub, etc are sliced along the chunk dim

        Return pooled output from BERT because that's all we need
        """
        X_sub = tf.compat.v1.Print(X_sub, ["X_sub", X_sub, tf.shape(X_sub)], summarize=-1)
        # To fit the interface of finetune we are going to compute the mask and type id at runtime.
        delimiters = tf.cast(tf.equal(X_sub, encoder.delimiter_token), tf.int32)
        # Appears to be 0 where you have tokens and 1 where you have padding
        # [batch_size, sequence_length]
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
        # Appears to be 1 where you have tokens and 0 where you have padding
        # [batch_size, sequence_length]
        mask = tf.sequence_mask(lengths, maxlen=seq_length, dtype=tf.float32)
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
        return bert.get_pooled_output()

    with tf.compat.v1.variable_scope("model/featurizer", reuse=reuse):
        # Get pooled output across chunk dim using map_fn
        # [num_chunks / batch_scaling, batch_size * batch_scaling, hidden_size]
        # pooled_output = tf.map_fn(lambda inp: bert_wrapper(inp[0], inp[1], inp[2]), (X, mask, token_type_ids))
        pooled_output = tf.map_fn(lambda inp: bert_wrapper(inp), X, dtype=tf.float32)
        pooled_output = tf.compat.v1.Print(pooled_output, ["pooled_output", tf.shape(pooled_output)], summarize=-1)

        # Reshape output to remove batch_scaling factor
        # [num_chunks, batch_size, hidden_size]
        pool_shape = tf.shape(input=pooled_output)
        pooled_output = tf.reshape(pooled_output,
                                   [pool_shape[0]*batch_scaling,
                                    tf.cast(pool_shape[1]/batch_scaling, dtype=tf.int32), pool_shape[2]])
        pooled_output = tf.compat.v1.Print(pooled_output, ["pooled_rs", tf.shape(pooled_output)], summarize=-1)

        # TODO In original featurizer, there's a reshape commented out below from pooled_output to features
        # Appeared to not change dims [batch_size, hidden_size], so not sure what it does
        # features = tf.reshape(
        #     pooled_output,
        #     shape=tf.concat((initial_shape[:-1], [config.n_embed]), 0),
        # )

        # Reduce across chunk dim with aggregation operation
        # Defaulting to mean now, TODO implement other aggregation options
        features = tf.reduce_mean(pooled_output, axis=0)
        features = tf.compat.v1.Print(features, ["features", tf.shape(features)], summarize=-1)

        output_state = {
            "features": features,
            # "lengths": lengths,
            # "eos_idx": eos_idx,
            "sequence_features": None,
        }
        if config.num_layers_trained == 0:
            output_state = {k: tf.stop_gradient(v) for k, v in output_state.items()}

        return output_state


layoutlm_featurizer = functools.partial(bert_featurizer, underlying_model=LayoutLMModel)

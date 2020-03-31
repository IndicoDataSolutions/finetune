import os
import sys

import tensorflow as tf
from finetune.util.shapes import lengths_from_eos_idx, merge_leading_dims
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder
from finetune.base_models.bert.modeling import BertConfig, BertModel
from finetune.nn.target_blocks import smooth_pos_attn
from finetune.errors import FinetuneError
from finetune.nn.auxiliary import dense_with_custom_init
from finetune.util.metrics import read_eval_metrics


def get_improvement_decay(total_num_steps, eval_dir, eval_freq, train):
    def py_has_improved():
        if not train:
            return False
        eval_results = {
            k: v for k, v in
            read_eval_metrics(eval_dir).items()
            if "loss" in v # Filter out anything in eval that does not have a loss value                                                               
        }
        if len(eval_results) == 0:
            return False
        most_recent_eval = max(eval_results.items(), key=lambda x: x[0])  # last steps.
        best_eval = min(eval_results.items(), key=lambda x: x[1]["loss"])  # lowest_loss
        if most_recent_eval == best_eval:
            return True
        return False
    
    has_improved = tf.cast(
        tf.cond(
            tf.equal((tf.train.get_global_step() - 1) % eval_freq, 0),
            true_fn=lambda: tf.py_function(func=py_has_improved, inp=[], Tout=tf.bool),
            false_fn=lambda: False
        ),
        tf.float32
    )
    decay = tf.get_variable("decay", shape=[], dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(0.0))
    decay_op = tf.assign(decay, get_decay_for_half(total_num_steps) * has_improved + decay * (1 - has_improved))
    with tf.control_dependencies([decay_op]):
        decay = tf.identity(decay)
    return decay       

def get_decay_for_half(total_num_steps):
    return tf.minimum(tf.cast(tf.train.get_global_step(), tf.float32) / (total_num_steps / 2), 1.0)

def non_normed_dropout(x, rate, noise_shape=None, placeholder=None):
    if placeholder is None:
        placeholder = 0.0
    if noise_shape is None:
        noise_shape = tf.shape(x)
    dropped = tf.cast(tf.random.uniform(shape=noise_shape) > rate, tf.float32)
    return dropped * x + (1.0 - dropped) * placeholder

def pos_mask_dropout_zero(train, rate):
    def proc(x):
        return non_normed_dropout(x, rate, noise_shape=[tf.shape(x)[0], 1])
    return proc

def pos_mask_dropout_placeholder(train, rate):
    def proc(x):
	return non_normed_dropout(
            x, rate, noise_shape=[tf.shape(x)[0], 1],
            placeholder=tf.get_variable("pos_placeholder", shape=[1, x.get_shape().as_list()[1]], dtype=tf.float32)
        )
    return proc

def get_pos_embedding_transform(pos_removal_mode, pos_decay_mode, train, total_num_steps, eval_freq, eval_dir):
    assert total_num_steps is not None
    if pos_removal_mode is None:
        return None
    
    if pos_decay_mode.lower() == "fixed":
        rate = 1.0
    elif pos_decay_mode.lower() == "step":
        rate = get_improvement_decay(total_num_steps, eval_dir, eval_freq, train)
    elif pos_decay_mode.lower() == "smooth":
        rate = get_decay_for_half(total_num_steps)
    tf.summary.scalar("pos_decay", rate)
    
    if pos_removal_mode.lower() == "zero_out":
        return pos_mask_dropout_zero(train, rate):
    elif pos_removal_mode.lower() == "placeholder":    
        return pos_mask_dropout_placeholder(train, rate)
    
    raise ValueError("pos_removal_mode {} not recognised".format(pos_removal_mode))


def bert_featurizer(
    X,
    encoder,
    config,
    train=False,
    reuse=None,
    context=None,
    total_num_steps=None,
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
    is_roberta = issubclass(config.base_model.encoder, RoBERTaEncoder)
    model_filename = config.base_model_path.rpartition('/')[-1]
    is_roberta_v1 = is_roberta and model_filename in ("roberta-model-sm.jl", "roberta-model-lg.jl")
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
        low_memory_mode=config.low_memory_mode,
        context_dim = config.context_dim,
        n_context_embed_per_channel = config.n_context_embed_per_channel,
        use_auxiliary_info=config.use_auxiliary_info and config.sidekick and not config.mlm_baseline,
        n_layers_with_aux=config.n_layers_with_aux,
        pos_injection=config.pos_injection,
        use_position_embeddings=config.use_reading_order_position,
        pos_embedding_transform=get_pos_embedding_transform(
            config.pos_embedding_transform,
            train,
            total_num_steps,
            config.val_interval,
            None,#os.path.join(config.tensorboard_folder, "eval")
        )
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
            roberta=is_roberta,
            context=context if not config.mlm_baseline else None
        )

        embed_weights = bert.get_embedding_table()

        if context is not None and config.sidekick:
            n_embed = config.n_embed + config.n_context_embed_per_channel * config.context_dim
        else:
            n_embed = config.n_embed

        features = tf.reshape(
            bert.get_pooled_output(),
            shape=tf.concat((initial_shape[:-2], [n_embed]), 0),
        )
        sequence_features = tf.reshape(
            bert.get_sequence_output(),
            shape=tf.concat((initial_shape[:-1], [n_embed]), 0),
        )

        # baseline just projects back to config.n_embed
        if context is not None and config.mlm_baseline:
            with tf.variable_scope('context'):
                # print('sequence_features', sequence_features)
                # print('context', context)
                sequence_features = tf.concat((sequence_features, context), -1)
                features = tf.concat((features, tf.reduce_mean(context, 1)), -1)
                print('sequence_features after concat', sequence_features)
                print('sequence_features merged', merge_leading_dims(sequence_features, 2))
                # sequence_features = tf.Print(sequence_features, [sequence_features], summarize=1000)
                # tf.print(sequence_features, output_stream=sys.stderr)
                pos_embed = config.n_context_embed_per_channel * config.context_dim
                # import ipdb; ipdb.set_trace()
                sequence_features = dense_with_custom_init(
                    merge_leading_dims(sequence_features, 2), config.n_embed, activation=None, name='seq_feats_proj',
                    kernel_initializer=None, custom=True, pos_embed=pos_embed, proj_type='downward_identity')
                sequence_features = tf.reshape(sequence_features, tf.concat((initial_shape[:-1], [config.n_embed]), 0))
                # sequence_features = tf.Print(sequence_features, [sequence_features], summarize=1000)
                features = dense_with_custom_init(
                    features, config.n_embed, activation=None, kernel_initializer=None,
                    custom=True, pos_embed=pos_embed, name='feats_proj', proj_type='downward_identity'
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

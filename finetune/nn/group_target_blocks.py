import math
import functools
import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood
from scipy.optimize import linear_sum_assignment

from finetune.base_models.gpt.featurizer import attn, dropout, norm
from finetune.util.shapes import shape_list, merge_leading_dims
from finetune.optimizers.recompute_grads import recompute_grad
from finetune.errors import FinetuneError
from finetune.nn.activations import act_fns
from finetune.nn.nn_utils import norm
from finetune.nn.target_blocks import sequence_labeler
from tensorflow.python.framework import function

from finetune.base_models.bert.modeling import (
    attention_layer,
    dropout,
    create_initializer,
    layer_norm,
)


def multi_crf_group_labeler(
    hidden,
    targets,
    n_targets,
    config,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=False,
    **kwargs
):
    """
    Multi CRF group tagging model. Takes two sets of targets - one for normal
    tagging and one for group tagging. Learns a CRF for each set of targets.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the NER and group targets. [batch_size, 2, sequence_length]
    :param n_targets: A python int containing the number of NER classes.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": Un-normalized log probabilties for NER and group predictions, has
        shape of [2, batch_size, sequence_length]
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.compat.v1.variable_scope("multi-crf-group", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed

        def seq_lab_internal(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, n_targets)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [n_targets]], 0)
            )
            return logits

        def group_seq_lab_internal(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, 3)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [3]], 0)
            )
            return logits

        with tf.compat.v1.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )
            logits = seq_lab_internal(hidden)
            logits = tf.cast(logits, tf.float32)  # always run the crf in float32
        with tf.compat.v1.variable_scope("group_seq_lab_attn"):
            if config.low_memory_mode and train:
                group_seq_lab_internal = recompute_grad(
                    group_seq_lab_internal, use_entire_scope=True
                )
            group_logits = group_seq_lab_internal(hidden)
            group_logits = tf.cast(group_logits, tf.float32)

        loss = 0.0

        default_lengths = tf.shape(input=hidden)[1] * tf.ones(
            tf.shape(input=hidden)[0], dtype=tf.int32
        )
        if lengths is None:
            lengths = default_lengths

        class_weights = kwargs.get("class_weights")

        with tf.device("CPU:0" if train else logits.device):
            if class_weights is not None and train:
                class_weights = tf.reshape(class_weights, [1, 1, -1])
                one_hot_class_weights = class_weights * tf.one_hot(
                    targets[:, 0, :], depth=n_targets
                )
                per_token_weights = tf.reduce_sum(
                    input_tensor=one_hot_class_weights, axis=-1, keepdims=True
                )
                logits = class_reweighted_grad(logits, per_token_weights)

            transition_params = tf.cast(
                tf.compat.v1.get_variable(
                    "Transition_matrix", shape=[n_targets, n_targets]
                ),
                tf.float32,
            )
            group_transition_params = tf.cast(
                tf.compat.v1.get_variable(
                    "Group_transition_matrix", shape=[3, 3]
                ),
                tf.float32,
            )
            if targets is not None:
                if use_crf:
                    ner_loss, _ = crf_log_likelihood(
                        logits,
                        targets[:, 0, :],
                        lengths,
                        transition_params=transition_params,
                    )
                    group_loss, _ = crf_log_likelihood(
                        group_logits,
                        targets[:, 1, :],
                        lengths,
                        transition_params=group_transition_params,
                    )
                    ner_loss = tf.reduce_mean(ner_loss * -1)
                    group_loss = tf.reduce_mean(group_loss * -1)
                else:
                    weights = tf.math.divide_no_nan(
                        tf.sequence_mask(
                            lengths,
                            maxlen=tf.shape(input=targets)[2],
                            dtype=tf.float32,
                        ),
                        tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                    )
                    ner_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                        targets[:, 0, :], logits, weights=weights
                    )
                    ner_loss = tf.reduce_mean(ner_loss)
                    group_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                        targets[:, 1, :], group_logits, weights=weights
                    )
                    group_loss = tf.reduce_mean(group_loss)
                scaled_ner_loss = config.crf_seq_loss_weight * ner_loss
                scaled_group_loss = config.crf_group_loss_weight * group_loss
                loss = scaled_ner_loss + scaled_group_loss

                tf.compat.v1.summary.scalar("Sequence Loss", ner_loss)
                tf.compat.v1.summary.scalar("Group Loss", group_loss)
                tf.compat.v1.summary.scalar("Scaled Sequence Loss", scaled_ner_loss)
                tf.compat.v1.summary.scalar("Scaled Group Loss", scaled_group_loss)

        return {
            "logits": [logits, group_logits],
            "losses": loss,
            "predict_params": {
                "transition_matrix": transition_params,
                "group_transition_matrix": group_transition_params,
                "sequence_length": lengths,
            },
        }

def multi_logit_group_labeler (
    hidden,
    targets,
    n_targets,
    config,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=False,
    **kwargs
):
    """
    Mult-logit CRF group tagging model. Produces a set of logits for NER tags
    and group tags, then broadcasts them to create the final predictions.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the sequence labeling targets. [batch_size, sequence_length]
    :param n_targets: A python int containing the number of total number of classes (NER Classes * 3)
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distribution is not sufficient and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    assert n_targets % 3 == 0, f"{n_targets} classes not divisible by 3!"
    with tf.compat.v1.variable_scope("multi-logit-group", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed

        def seq_lab_internal(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, n_targets // 3)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [n_targets // 3]], 0)
            )
            return logits

        def group_seq_lab_internal(hidden):
            # Produce 3 outputs: start group, in group, outside of group
            flat_logits = tf.compat.v1.layers.dense(hidden, 3)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [3]], 0)
            )
            return logits

        with tf.compat.v1.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )
            ner_logits = seq_lab_internal(hidden)
            ner_logits = tf.cast(ner_logits, tf.float32)  # always run the crf in float32
        with tf.compat.v1.variable_scope("group_seq_lab_attn"):
            if config.low_memory_mode and train:
                group_seq_lab_internal = recompute_grad(
                    group_seq_lab_internal, use_entire_scope=True
                )
            group_logits = group_seq_lab_internal(hidden)
            group_logits = tf.cast(group_logits, tf.float32)

        # Broadcast probabilities to make [batch, seq_len, n_classes] matrix
        # [batch, seq_len, n_classes / 3, 1] * [batch, seq_len, 1, 3] = 
        # [batch, seq_len, n_classes / 3, 3]
        logits = tf.expand_dims(ner_logits, 3) + tf.expand_dims(group_logits, 2)
        # Reshape down to [batch, seq_len, n_classes]
        final_shape = tf.concat((tf.shape(hidden)[:2], [n_targets]), 0)
        logits = tf.reshape(logits, final_shape)
        # Note, in order for loss to work correctly the targets must be in the
        # form [AA-TAG1, BB-TAG1, CC-TAG1, AA-TAG2, BB-TAG2, CC-TAG2, AA-TAG3 ...]
        # where tags are grouped together and always in the same order of
        # prefixes, as this is the form the logits will be produced in via broadcasting

        loss = 0.0

        default_lengths = tf.shape(input=hidden)[1] * tf.ones(
            tf.shape(input=hidden)[0], dtype=tf.int32
        )
        if lengths is None:
            lengths = default_lengths

        class_weights = kwargs.get("class_weights")

        with tf.device("CPU:0" if train else logits.device):
            if class_weights is not None and train:
                class_weights = tf.reshape(class_weights, [1, 1, -1])
                one_hot_class_weights = class_weights * tf.one_hot(
                    targets, depth=n_targets
                )
                per_token_weights = tf.reduce_sum(
                    input_tensor=one_hot_class_weights, axis=-1, keepdims=True
                )
                logits = class_reweighted_grad(logits, per_token_weights)

            transition_params = tf.cast(
                tf.compat.v1.get_variable(
                    "Transition_matrix", shape=[n_targets, n_targets]
                ),
                tf.float32,
            )
            if targets is not None:
                if use_crf:
                    log_likelihood, _ = crf_log_likelihood(
                        logits,
                        targets,
                        lengths,
                        transition_params=transition_params,
                    )
                    loss = -log_likelihood
                else:
                    weights = tf.math.divide_no_nan(
                        tf.sequence_mask(
                            lengths,
                            maxlen=tf.shape(input=targets)[1],
                            dtype=tf.float32,
                        ),
                        tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                    )
                    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                        targets, logits, weights=weights
                    )

        return {
            "logits": logits,
            "losses": loss,
            "predict_params": {
                "transition_matrix": transition_params,
                "sequence_length": lengths,
            },
        }

def bros_decoder(
    hidden,
    targets,
    n_targets,
    config,
    train=False,
    reuse=None,
    lengths=None,
    **kwargs
):
    """
    A BROS decoder.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The targets. Contains start token and next token labels. [batch_size, 2, sequence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": Dictionary containing "start_token_logits" and "next_token_logits"
        "losses": Combined start token and next token loss
    """
    with tf.compat.v1.variable_scope("bros-decoder", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed
        hidden_size = config.relation_hidden_size

        def get_out_logits(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, 2)
            logits_shape = tf.concat([tf.shape(hidden)[:2], [2]], 0)
            logits = tf.reshape(flat_logits, logits_shape)
            return logits
        def get_out_hidden(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, hidden_size)
            logits_shape = tf.concat([tf.shape(hidden)[:2], [hidden_size]], 0)
            logits = tf.reshape(flat_logits, logits_shape)
            return logits

        with tf.compat.v1.variable_scope("start_token_logits"):
            # [Batch Size, Sequence Length, 2]
            if config.low_memory_mode and train:
                get_out_logits = recompute_grad(get_out_logits, use_entire_scope=True)
            start_token_logits = get_out_logits(hidden)
            start_token_logits = tf.cast(start_token_logits, tf.float32)
        with tf.compat.v1.variable_scope("start_token_hidden"):
            if config.low_memory_mode and train:
                get_out_hidden = recompute_grad(get_out_hidden, use_entire_scope=True)
            # [Batch Size, Sequence Length, Hidden Size]
            start_token_hidden = get_out_hidden(hidden)
            start_token_hidden = tf.cast(start_token_hidden, tf.float32)
        with tf.compat.v1.variable_scope("next_token_hidden"):
            if config.low_memory_mode and train:
                get_out_hidden = recompute_grad(get_out_hidden, use_entire_scope=True)
            # [Batch Size, Sequence Length, Hidden Size]
            next_token_hidden = get_out_hidden(hidden)
            next_token_hidden = tf.cast(next_token_hidden, tf.float32)

            no_next_hidden = tf.cast(
                tf.compat.v1.get_variable(
                    "no_next_hidden", shape=[hidden_size]
                ),
                tf.float32,
            )
            no_next_hidden = tf.reshape(no_next_hidden, (1, 1, hidden_size))
            no_next_hidden = tf.repeat(no_next_hidden, tf.shape(hidden)[0], axis=0)
            # [Batch Size, Sequence Length + 1, Hidden Size]
            # Note: The no relation embedding comes first

            next_token_hidden = tf.concat((no_next_hidden, next_token_hidden), axis=1)
        with tf.compat.v1.variable_scope("next_token_logits"):
            # [Batch Size, Sequence Length, Sequence Legth + 1]
            next_token_logits = tf.matmul(
                start_token_hidden,
                next_token_hidden,
                transpose_b=True
            )

        if lengths is None:
            lengths = tf.shape(input=hidden)[1] * tf.ones(
                tf.shape(input=hidden)[0], dtype=tf.int32
            )

        loss = 0.0
        # class_weights = kwargs.get("class_weights")
        with tf.device("CPU:0" if train else start_token_logits.device):
            # if class_weights is not None and train:
            #     class_weights = tf.reshape(class_weights, [1, 1, -1])
            #     one_hot_class_weights = class_weights * tf.one_hot(
            #         targets, depth=n_targets
            #     )
            #     per_token_weights = tf.reduce_sum(
            #         input_tensor=one_hot_class_weights, axis=-1, keepdims=True
            #     )
            #     logits = class_reweighted_grad(logits, per_token_weights)

            if targets is not None:
                weights = tf.math.divide_no_nan(
                    tf.sequence_mask(
                        lengths,
                        maxlen=tf.shape(input=targets)[2],
                        dtype=tf.float32,
                    ),
                    tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                )
                start_token_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                    targets[:, 0, :], start_token_logits, weights=weights
                )
                next_token_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                    targets[:, 1, :], next_token_logits, weights=weights
                )
                scaled_start_token_loss = config.start_token_loss_weight * start_token_loss
                scaled_next_token_loss = config.next_token_loss_weight * next_token_loss
                loss = scaled_start_token_loss + scaled_next_token_loss

                tf.compat.v1.summary.scalar("Start Token Loss", start_token_loss)
                tf.compat.v1.summary.scalar("Next Token Loss", next_token_loss)
                tf.compat.v1.summary.scalar("Scaled Start Token Loss", scaled_start_token_loss)
                tf.compat.v1.summary.scalar("Scaled Next Token Loss", scaled_next_token_loss)

        return {
            "logits": {
                "start_token_logits": start_token_logits,
                "next_token_logits": next_token_logits,
            },
            "losses": loss,
        }

def joint_bros_decoder(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=True,
    **kwargs
):
    """
    A target block that calls both the sequence labeling target block and the
    BROS decoder target block. See each respective function for further details

    :param targets: The targets. The [:, 0, :] is assumed to contain the sequence labeling targets,
    and the rest ([:, 1:, :]), is expected to contain the BROS labels. [batch_size, 3, sequence_length]
    :return: dict containing:
        "logits": Dictionary containing "ner_logits", "start_token_logits" and "next_token_logits"
        "losses": Combined sequence labeling and bros loss
    """
    # Unpack the targets
    bros_targets, seq_targets = None, None
    if targets is not None:
        bros_targets = targets[:, 1:, :]
        seq_targets = targets[:, 0, :]
        
    bros_dict = bros_decoder(
        hidden,
        bros_targets,
        n_targets,
        config,
        train=train,
        reuse=reuse,
        lengths=lengths,
        **kwargs
    )
    seq_dict = sequence_labeler(
        hidden,
        seq_targets,
        n_targets,
        config,
        pad_id,
        multilabel=False,
        train=train,
        reuse=reuse,
        lengths=lengths,
        use_crf=use_crf,
        **kwargs
    )

    seq_loss = tf.reduce_mean(seq_dict["losses"])
    scaled_seq_loss = config.seq_loss_weight * seq_loss
    scaled_group_loss = config.group_loss_weight * bros_dict["losses"]
    tf.compat.v1.summary.scalar("Sequence Loss", seq_loss)
    tf.compat.v1.summary.scalar("Group Loss", bros_dict["losses"])
    tf.compat.v1.summary.scalar("Scaled Sequence Loss", scaled_seq_loss)
    tf.compat.v1.summary.scalar("Scaled Group Loss", scaled_group_loss)

    return {
        "logits": {
            "ner_logits": seq_dict["logits"],
            "start_token_logits": bros_dict["logits"]["start_token_logits"],
            "next_token_logits": bros_dict["logits"]["next_token_logits"],
        },
        "losses": scaled_seq_loss + scaled_group_loss,
        "predict_params": {
            "sequence_length": lengths,
            "transition_matrix": seq_dict["predict_params"]["transition_matrix"],
        },
    }

def token_relation_decoder(
    hidden,
    targets,
    n_targets,
    config,
    train=False,
    reuse=None,
    lengths=None,
    **kwargs
):
    """
    A token level relation decoder.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The targest. Contains an entity mask and a relation matrix. [batch_size, 2, sequence_length, sequence_length]
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": Un-normalized relation matrix logits. [batch_size, sequence_length, sequence_length]
        "losses": Masked BCE loss.
    """
    with tf.compat.v1.variable_scope("token-relation-decoder", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed
        hidden_size = config.relation_hidden_size

        def get_out_hidden(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, hidden_size)
            logits_shape = tf.concat([tf.shape(hidden)[:2], [hidden_size]], 0)
            logits = tf.reshape(flat_logits, logits_shape)
            return logits

        with tf.compat.v1.variable_scope("token_hidden"):
            if config.low_memory_mode and train:
                get_out_hidden = recompute_grad(get_out_hidden, use_entire_scope=True)
            # [Batch Size, Sequence Length, Hidden Size]
            token_hidden = get_out_hidden(hidden)
            token_hidden = tf.cast(token_hidden, tf.float32)
        with tf.compat.v1.variable_scope("logits"):
            # [Batch Size, Sequence Length, Sequence Legth]
            logits = tf.matmul(
                token_hidden,
                token_hidden,
                transpose_b=True
            )
        if lengths is None:
            lengths = tf.shape(input=hidden)[1] * tf.ones(
                tf.shape(input=hidden)[0], dtype=tf.int32
            )

        loss = 0.0
        with tf.device("CPU:0" if train else logits.device):
            if targets is not None:
                mask, targets = targets[:, 0, :], targets[:, 1, :]
                # weights = tf.math.divide_no_nan(
                #     tf.sequence_mask(
                #         lengths,
                #         maxlen=tf.shape(input=targets)[1],
                #         dtype=tf.float32, #     ),
                #     tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                # )
                targets *= mask
                logits *= tf.cast(mask, tf.float32)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    targets, logits, from_logits=True,
                ))
 
        return {
            "logits": logits,
            "losses": loss,
            "predict_params": {
                "sequence_length": lengths,
            },
        }

def joint_token_relation_decoder(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=True,
    **kwargs
):
    """
    A target block that calls both the sequence labeling target block and the
    token relation decoder target block. See each respective function for further details

    :param targets: The targets. The [:, 0, :, :] is assumed to contain a
    padded matrix with the sequence labeling targets, and the rest ([:, 1:, :, :]),
    is expected to contain the BROS labels. [batch_size, 3, sequence_length, sequence_length]
    :return: dict containing:
        "logits": Dictionary containing "ner_logits" [batch_size, seq_len]
        and "group_logits" [batch_size, seq_len, seq_len]
        "losses": Combined sequence labeling and token relation loss
    """
    token_relation_targets, seq_targets = None, None
    if targets is not None:
        token_relation_targets = targets[:, 1:, :, :]
        seq_targets = targets[:, 0, :, :]
        seq_targets = seq_targets[:, 0, :]
        
    token_relation_dict = token_relation_decoder(
        hidden,
        token_relation_targets,
        n_targets,
        config,
        train=train,
        reuse=reuse,
        lengths=lengths,
        **kwargs
    )
    seq_dict = sequence_labeler(
        hidden,
        seq_targets,
        n_targets,
        config,
        pad_id,
        multilabel=False,
        train=train,
        reuse=reuse,
        lengths=lengths,
        use_crf=use_crf,
        **kwargs
    )

    seq_loss = tf.reduce_mean(seq_dict["losses"])
    scaled_seq_loss = config.seq_loss_weight * seq_loss
    scaled_group_loss = config.group_loss_weight * token_relation_dict["losses"]
    tf.compat.v1.summary.scalar("Sequence Loss", seq_loss)
    tf.compat.v1.summary.scalar("Group Loss", token_relation_dict["losses"])
    tf.compat.v1.summary.scalar("Scaled Sequence Loss", scaled_seq_loss)
    tf.compat.v1.summary.scalar("Scaled Group Loss", scaled_group_loss)

    return {
        "logits": {
            "ner_logits": seq_dict["logits"],
            "group_logits": token_relation_dict["logits"],
        },
        "losses": scaled_seq_loss + scaled_group_loss,
        "predict_params": {
            "sequence_length": lengths,
            "transition_matrix": seq_dict["predict_params"]["transition_matrix"],
        },
    }

@tf.function
def tf_linear_sum_assignment(cost_matrix):
    return tf.numpy_function(func=linear_sum_assignment,inp=[cost_matrix],Tout=[tf.int64,tf.int64])

def decoder_block(
    layer_input,
    encoder_output,
    attention_head_size,
    batch_size,
    seq_length,
    encoder_length,
    cross_attention_mask=None,
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    intermediate_act_fn=gelu,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    initializer_range=0.02,
    drop_self_attention=False,
):
    """
    Taken from base_models.bert.modeling.full_block()
    Applies cross attention in addition to the blocks normal operations
    """

    def _attention(from_tensor, to_tensor, mask, from_len, to_len):
        with tf.compat.v1.variable_scope("attention"):
            attention_output = attention_layer(
                from_tensor=from_tensor,
                to_tensor=to_tensor,
                attention_mask=mask,
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                do_return_2d_tensor=True,
                batch_size=batch_size,
                from_seq_length=from_len,
                to_seq_length=to_len,
            )
        with tf.compat.v1.variable_scope("output"):
            attention_output = tf.compat.v1.layers.dense(
                attention_output,
                hidden_size,
                kernel_initializer=create_initializer(initializer_range))
            attention_output = dropout(attention_output, hidden_dropout_prob)
            attention_output = layer_norm(attention_output + layer_input)
        return attention_output

    if self.drop_self_attention:
        attention_output = layer_input
    else:
        with tf.compat.v1.variable_scope("self-attention"):
            attention_output = _attention(
                layer_input, layer_input, None, seq_len, seq_len
            )
    with tf.compat.v1.variable_scope("cross-attention"):
        attention_output = _attention(
            attention_output, encoder_output, cross_attention_mask,
            seq_len, encoder_length
        )

    # The activation is only applied to the "intermediate" hidden layer.
    with tf.compat.v1.variable_scope("intermediate"):
        intermediate_output = tf.compat.v1.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

    # Down-project back to `hidden_size` then add the residual.
    with tf.compat.v1.variable_scope("output"):
        layer_output = tf.compat.v1.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
    return layer_output

def group_relation_decoder(
    hidden,
    targets,
    n_groups,
    config,
    train=False,
    reuse=None,
    lengths=None,
    hidden_size=768,
    num_attention_heads=12,
    n_layers=3,
    query_size=256,
    **kwargs
):
    with tf.compat.v1.variable_scope("seq2seq-decoder"):
        batch_size, encoder_length, encoder_hidden = tf.shape(hidden)

        # Group embeddings
        group_embeddings = tf.cast(
            tf.compat.v1.get_variable(
                "group_embeddings", shape=[n_groups, hidden_size]
            ),
            tf.float32,
        )
        # Duplicate group embeddings along batch dimension
        group_embeddings = tf.tile(group_embeddings[None, :, :], [batch_size, 1, 1])
        # Reshape to 2D for attention calculations
        group_embeddings = tf.reshape(group_embeddings, [-1, hidden_size])
        hidden = tf.reshape(hidden, [-1, encoder_hidden])

        # Decoder blocks
        # Modified version of base_models.bert.modeling.transformer_model()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        attention_head_size = int(hidden_size / num_attention_heads)

        # Build cross attention mask
        # [batch_size, 1, seq_len]
        encoder_mask = tf.sequence_mask(
            lengths, encoder_length, dtype=tf.int32
        )[:, None, :]
        # [batch_size, n_groups, 1]
        decoder_mask = tf.ones([batch_size, n_groups, 1], dtype=tf.int32)
        # [batch_size, n_groups, seq_len]
        mask = decoder_mask * encoder_mask

        layer_input = group_embeddings
        for layer_idx in range(n_layers):
            with tf.compat.v1.variable_scope(f"decoder-block-{layer_idx}"):
                block_fn = functools.partial(
                    decoder_block,
                    encoder_output=hidden,
                    attention_head_size=attention_head_size,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    encoder_length=encoder_length,
                    cross_attention_mask=mask,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    intermediate_act_fn=intermediate_act_fn,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    initializer_range=initializer_range,
                    drop_self_attention=(layer_idx == 0)
                )
                if config.low_memory_mode:
                    block_fn = recompute_grad(block_fn, use_entire_scope=True)
                layer_input = block_fn(layer_input)
        attention_output = tf.reshape(
            layer_input, [batch_size, n_groups, hidden_size]
        )

        # Output logits
        with tf.compat.v1.variable_scope("logits"):
            # [batch_size, n_groups, query_size]
            group_queries = tf.compat.v1.layers.dense(attention_output, query_size)
            # [batch_size, seq_len, query_size]
            token_keys = tf.compat.v1.layers.dense(hidden, query_size)
            # [batch_size, n_groups, seq_len]
            logits = tf.matmul(group_queries, token_keys, transpose_b=True)
            logits *= 1.0 / math.sqrt(float(query_size))
            probs = tf.sigmoid(logits)

        loss = 0.0
        if targets:
            # Hungarian Algorithm
            with tf.compat.v1.variable_scope("hungarian-algorithm"):
                # Cost matrix - get the sum of probailities of tokens in group
                # [batch_size, n_groups, 1, seq_len]
                cost_probs = probs[:, :, None, :]
                # [batch_size, 1, n_groups, seq_len]
                cost_targets = targets[:, None, :, :]
                # [batch_size, n_groups, n_groups, seq_len]
                costs = probs * targets
                # [batch_size, n_groups, n_groups]
                costs = tf.reduce_sum(costs, axis=-1)

                # Normalize cost matrix
                # [batch_size, 1, n_groups]
                n_tokens_per_group = tf.reduce_sum(targets, axis=-1)[:, None, :]
                # [batch_size, n_groups, n_groups]
                costs /= n_tokens_per_group

                # Run Hungarian Algorithm 
                # [batch_size, 2, n_groups]
                idxs = tf.map_fn(tf_linear_sum_assignment, costs)

                # Reorder targets accordingly
                # [batch_size, n_groups]
                target_idxs = idxs[:, 1, :]
                # [batch_size, n_groups, seq_len]
                targets = tf.gather(targets, target_idxs, batch_dim=1)

            # Loss calculation
            with tf.compat.v1.variable_scope("loss"):
                # Averages over last dimension, so [batch_size, n_groups]
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    targets, logits, from_logits=True,
                ))

    return {
        "logits": logits,
        "losses": loss,
        "predict_params": {
            "probs": probs
        },
    }

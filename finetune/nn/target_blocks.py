import functools
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood

from finetune.base_models.gpt.featurizer import attn, dropout, norm
from finetune.util.shapes import shape_list, merge_leading_dims
from finetune.optimizers.recompute_grads import recompute_grad
from finetune.optimizers.tsa_schedules import get_tsa_threshold, tsa_loss
from finetune.errors import FinetuneError
from finetune.nn.activations import act_fns
from finetune.nn.nn_utils import norm

def perceptron(x, ny, config, w_init=None, b_init=None):
    """
    A very standard linear Perceptron model.
    :param x: Input tensor.
    :param ny: Number of outputs.
    :param config: A config object.
    :param w_init: Weight initializer.
    :param b_init: Bias initializer.
    :return: The output of the perceptron model.
    """
    w_init = w_init or tf.random_normal_initializer(stddev=config.weight_stddev)
    b_init = b_init or tf.constant_initializer(0)

    with tf.variable_scope('perceptron'):
        nx = config.n_embed
        if config.use_auxiliary_info:
            nx += config.n_context_embed
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = shape_list(sequence_tensor)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1]
    )
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def masked_language_model(*, X, M, mlm_weights, mlm_positions, mlm_ids, embed_weights, hidden, config, reuse=None, train=False):
    X = merge_leading_dims(X, 3)
    M = merge_leading_dims(M, 2)
    hidden = merge_leading_dims(hidden, 3)
    batch, seq, _ = shape_list(X)
    with tf.variable_scope('model/masked-language-model'):
        gathered_hidden = gather_indexes(hidden, mlm_positions)
        final_proj = tf.layers.dense(
            gathered_hidden,
            units=config.n_embed,
            activation=act_fns[config.act_fn],
            kernel_initializer=tf.random_normal_initializer(stddev=config.weight_stddev),
            name='dense'
        )
        normed_proj = norm(final_proj, 'LayerNorm')
        n_vocab = shape_list(embed_weights)[0]
        output_bias = tf.get_variable(
            "output_bias",
            shape=[n_vocab],
            initializer=tf.zeros_initializer()
        )
        logits = tf.matmul(normed_proj, embed_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        
        mlm_ids = tf.reshape(mlm_ids, [-1])
        mlm_weights = tf.reshape(mlm_weights, [-1])

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(mlm_ids, depth=n_vocab, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(mlm_weights * per_example_loss)
        denominator = tf.reduce_sum(mlm_weights) + 1e-5
        mlm_loss = numerator / denominator

        return {
            "logits": logits,
            "losses": mlm_loss,
        }


def language_model(*, X, M, embed_weights, hidden, config, reuse=None, train=False):
    """
    A language model output and loss for the language modelling objective described in the original finetune paper.
    This language model uses weights that are tied to the input embedding.
    :param X: The raw token ids fed to the featurizer.
    :param M: A loss mask, with 1's where losses should be counted and 0's elsewhere.
    :param embed_weights: The word embedding matrix, normally the one returned by the featurizer.
    :param hidden: Output of the featurizer.
    :param config: A config object.
    :param reuse: A Flag passed through to the tf.variable_scope context manager.
    :return: A dict containing:
        logits: The un-normalised log-probabilities over each word in the vocabulary.
        loss: The masked language modelling loss.

    """
    X = merge_leading_dims(X, 3)
    M = merge_leading_dims(M, 2)
    hidden = merge_leading_dims(hidden, 3)

    batch, seq, _ = shape_list(X)
    vocab_size, hidden_dim = shape_list(embed_weights)

    with tf.variable_scope('model/language-model', reuse=reuse):
        # language model ignores last hidden state because we don't have a target
        lm_h = tf.reshape(hidden, [-1, config.n_embed])  # [batch, seq_len, embed] --> [batch * seq_len, embed]
        lm_logits = tf.matmul(lm_h, embed_weights, transpose_b=True)  # tied weights
        lm_logits = tf.cast(lm_logits, tf.float32)
        hidden_shape = tf.shape(hidden)
        logits = tf.reshape(lm_logits, shape=tf.concat([hidden_shape[:-1], [vocab_size]], axis=0))
        lm_logits_offset = tf.reshape(logits[:, :-1], [-1, vocab_size])
        
        lm_losses = tf.losses.sparse_softmax_cross_entropy(
            logits=lm_logits_offset,
            labels=tf.reshape(X[:, 1:, 0], [-1]),
            weights=tf.reshape(M[:, 1:], [-1])
        )

        perplexity = tf.reduce_sum(tf.exp(lm_losses) * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)

        return {
            "logits": logits,
            "losses": lm_losses,
            "perplexity": perplexity,
        }


def _apply_class_weight(losses, targets, class_weights=None):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = (
            tf.reduce_sum(class_weights * tf.to_float(targets), axis=1)
        )
        weights *= tf.to_float(tf.reduce_prod(tf.shape(weights))) / tf.reduce_sum(
            weights
        )
        losses *= tf.expand_dims(weights, 1)
    return losses


def _apply_multilabel_class_weight(losses, targets, class_weights=None):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = (
            # contribution of positive class
            class_weights * tf.to_float(targets) + 
            # contribution of negative class
            tf.ones_like(class_weights) * (1 - tf.to_float(targets))
        )
        weights *= tf.to_float(tf.reduce_prod(tf.shape(weights))) / tf.reduce_sum(
            weights
        )
        losses *= weights
    return losses


def classifier(hidden, targets, n_targets, config, train=False, reuse=None, **kwargs):
    """
    A simple linear classifier.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: One hot encoded target ids. [batch_size, n_classes]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param dropout_placeholder:
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The unnormalised log probabilities of each class.
        losses: The loss for the classifier.
    """
    with tf.variable_scope("classifier", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        clf_logits = perceptron(hidden, n_targets, config)
        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=clf_logits, labels=tf.stop_gradient(targets)
            )

            clf_losses = _apply_class_weight(
                clf_losses, targets, kwargs.get("class_weights")
            )

            # From Unsupervised Data Augmentation for Consistency Training, Xie et al. 2019
            if config.tsa_schedule:
                clf_logits, clf_losses = tsa_loss(
                    n_targets, config, clf_losses, clf_logits, targets
                )

        return {"logits": clf_logits, "losses": clf_losses}


def multi_choice_question(
    hidden, 
    targets, 
    n_targets, 
    config, 
    train=False, 
    reuse=None, 
    **kwargs
):
    with tf.variable_scope("model", reuse=reuse):
        if targets is not None:
            targets = tf.cast(targets, tf.int32)
        hidden = dropout(hidden, config.clf_p_drop, train)
        hidden = tf.unstack(hidden, num=n_targets, axis=1)
        hidden = tf.concat(hidden, axis=0)

        clf_out = perceptron(hidden, 1, config)
        clf_out = tf.split(clf_out, n_targets, axis=0)
        clf_out = tf.concat(clf_out, 1)

        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=clf_out, labels=tf.stop_gradient(targets)
            )

            clf_losses = _apply_class_weight(
                clf_losses, targets, kwargs.get("class_weights")
            )

        return {"logits": clf_out, "losses": clf_losses}


def multi_classifier(
    hidden, targets, n_targets, config, train=False, reuse=None, **kwargs
):
    """
    A simple linear classifier.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the sparse targets [batch_size, n_targets]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The unnormalised log probabilities of each class.
        losses: The loss for the classifier.
    """
    with tf.variable_scope("model", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        clf_logits = perceptron(hidden, n_targets, config)
        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=clf_logits, labels=tf.stop_gradient(targets)
            )
            clf_losses = _apply_multilabel_class_weight(
                clf_losses, targets, kwargs.get("class_weights")
            )
        return {"logits": clf_logits, "losses": clf_losses}


def regressor(hidden, targets, n_targets, config, train=False, reuse=None, **kwargs):
    """
    A simple linear regressor.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the regression targets. [batch_size]
    :param n_targets: A python int containing the number of outputs that the model should be learning to predict over.
    :param dropout_placeholder:
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The regression outputs.
        losses: L2 Loss for the regression targets.
    """
    with tf.variable_scope("regressor", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        outputs = perceptron(hidden, n_targets, config)
        if targets is None:
            loss = None
        else:
            if config.regression_loss.upper() == "L2":
                loss = tf.nn.l2_loss(outputs - targets)
            elif config.regression_loss.upper() == "L1":
                loss = tf.abs(outputs - targets)
            else:
                raise FinetuneError(
                    "regression_loss needs to be either L1 or L2, instead it is {}".format(
                        config.regression_loss
                    )
                )
        return {"logits": outputs, "losses": loss}


def ordinal_regressor(
    hidden,
    targets,
    n_targets,
    config,
    shared_threshold_weights=True,
    train=False,
    reuse=None,
    **kwargs
):
    """
    Ordinal Regressor using all-threshold loss.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the regression targets (binary threshold values). [batch_size]
    :param n_targets: A python int containing the number of thresholds that the model should be learning to predict over.
    :param dropout_placeholder:
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The regression outputs.
        losses: All-threshold Loss for the regression targets.
    """
    with tf.variable_scope("ordinalregressor", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        if shared_threshold_weights:
            w_init = tf.random_normal_initializer(stddev=config.weight_stddev)
            b_init = tf.random_normal_initializer(0)
            nx = config.n_embed
            w = tf.get_variable("w", [nx, 1], initializer=w_init)
            b = tf.get_variable("b", [n_targets], initializer=b_init)
            logits = tf.matmul(hidden, w) + b
        else:
            logits = perceptron(hidden, n_targets, config)

        if targets is None:
            outputs = tf.sigmoid(logits)
            loss = None
        else:
            outputs = logits
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=tf.stop_gradient(targets)
            )
        return {"logits": outputs, "losses": loss}


def class_reweighting(class_weights):
    @tf.custom_gradient
    def custom_grad(logits):
        def grad(g):
            new_g = g * class_weights
            ratio = tf.norm(g) / tf.norm(new_g)
            return new_g * ratio

        return tf.identity(logits), grad

    return custom_grad


def sequence_labeler(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    multilabel=False,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=True,
    **kwargs
):
    """
    An Attention based sequence labeler model.

    In the case of unidirectional base models such as GPT this model takes the output of the pre-trained model,
    applies an additional randomly initialised multihead attention block, with residuals on top.
    The extra attention is not future masked to allow the model to label sequences based on context in both directions.
    The representations fed into this model are necessarily future masked because a language modelling loss is the
    original objective of the featurizer.

    For bidirectional base models we apply the crf model directly to the output of the base model.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the sequence labeling targets. [batch_size, sequence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
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
    with tf.variable_scope("sequence-labeler", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed
        if config.use_auxiliary_info:
            nx += config.n_context_embed

        def seq_lab_internal(hidden):
            if config.base_model.is_bidirectional:
                n = hidden
            else:
                attn_fn = functools.partial(
                    attn,
                    scope="seq_label_attn",
                    n_state=nx,
                    n_head=config.seq_num_heads,
                    resid_pdrop=config.resid_p_drop,
                    attn_pdrop=config.attn_p_drop,
                    train=train,
                    scale=False,
                    mask=False,
                )
                n = norm(attn_fn(hidden) + hidden, "seq_label_residual")
            
            flat_logits = tf.layers.dense(n, n_targets)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(hidden)[:2], [n_targets]], 0)
            )
            return logits

        with tf.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )
            logits = seq_lab_internal(hidden)

        loss = 0.0

        default_lengths = tf.shape(hidden)[1] * tf.ones(
            tf.shape(hidden)[0], dtype=tf.int32
        )
        if lengths is None:
            lengths = default_lengths
            
        class_weights = kwargs.get("class_weights")
        
        with tf.device("CPU:0" if train else logits.device):
            if multilabel:
                transition_params = []
                logits_individual = tf.unstack(logits, n_targets, axis=-1)
                if targets is not None:
                    targets_individual = tf.unstack(targets, n_targets, axis=-1)
                logits = []
                for i in range(n_targets):
                    transition_params.append(
                        tf.get_variable("Transition_matrix_{}".format(i), shape=[2, 2])
                    )
                    logits.append(
                        tf.stack(
                            (logits_individual[pad_id], logits_individual[i]), axis=-1
                        )
                    )
                    if targets is not None and i != pad_id:
                        if class_weights is not None:
                            is_pos_cls = tf.cast(targets_individual[i], dtype=tf.float32)
                            class_weight = tf.expand_dims(class_weights[i] * is_pos_cls + class_weights[pad_id] * (1.0 - is_pos_cls), -1)
                            logits_i = class_reweighting(class_weight)(logits[-1])
                        else:
                            logits_i = logits[i]
                        if use_crf:
                            loss -= crf_log_likelihood(
                                logits_i,
                                targets_individual[i],
                                lengths,
                                transition_params=transition_params[-1],
                            )[0]
                        else:
                            weights = tf.sequence_mask(
                                lengths, maxlen=tf.shape(targets_individual[i])[1], dtype=tf.float32
                            ) / tf.expand_dims(tf.cast(lengths, tf.float32), -1)
                            loss += tf.compat.v1.losses.sparse_softmax_cross_entropy(
                                targets_individual[i],
                                logits_i,
                                weights=weights
                            )
                logits = tf.stack(logits, axis=-1)
            else:
                if class_weights is not None and train:
                    class_weights = tf.reshape(class_weights, [1, 1, -1])
                    one_hot_class_weights = class_weights * tf.one_hot(targets, depth=n_targets)
                    per_token_weights = tf.reduce_sum(
                        one_hot_class_weights, axis=-1, keep_dims=True
                    )
                    logits = class_reweighting(per_token_weights)(logits)
                                                                                                          
                transition_params = tf.get_variable(
                    "Transition_matrix", shape=[n_targets, n_targets]
                )
                if targets is not None:
                    if use_crf:
                        log_likelihood, _ = crf_log_likelihood(
                            logits, targets, lengths, transition_params=transition_params
                        )
                        loss = -log_likelihood
                    else:
                        weights = tf.sequence_mask(
                            lengths, maxlen=tf.shape(targets)[1], dtype=tf.float32
                        ) / tf.expand_dims(tf.cast(lengths, tf.float32), -1)
                        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                            targets,
                            logits,
                            weights=weights
                        )

        return {
            "logits": logits,
            "losses": loss,
            "predict_params": {"transition_matrix": transition_params, "sequence_length": lengths},
        }


def association(
    hidden, lengths, targets, n_targets, config, train=False, reuse=None, **kwargs
):
    """
    An Attention based sequence labeler model with association.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param lengths: The number of non-padding tokens in the input.
    :param targets: A dict containing:
     'labels': The sequence labeling targets. [batch_size, sequence_length],
     'associations': A matrix of class ids for the associations [batch_size, sequence_length, seqence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distrobution is not sufficiant and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.variable_scope("sequence-labeler", reuse=reuse):
        nx = config.n_embed
        length = config.max_length
        num_associations = len(config.association_types) + 1

        def seq_lab_internal(hidden):
            attn_fn = functools.partial(
                attn,
                scope="seq_label_attn",
                n_state=nx,
                n_head=config.seq_num_heads,
                resid_pdrop=config.resid_p_drop,
                attn_pdrop=config.attn_p_drop,
                train=train,
                scale=False,
                mask=False,
                lengths=lengths,
            )
            n = norm(attn_fn(hidden) + hidden, "seq_label_residual")
            flat_logits = tf.layers.dense(n, n_targets)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(hidden)[:2], [n_targets]], 0)
            )

            association_head = tf.layers.dense(n, nx)
            association_head = tf.reshape(
                association_head, tf.concat([tf.shape(hidden)[:2], [nx]], 0)
            )

            a = tf.expand_dims(association_head, 1)
            b = tf.expand_dims(association_head, 2)

            features = tf.concat(
                [
                    a - b,
                    a * b,
                    tf.tile(a, [1, length, 1, 1]),
                    tf.tile(b, [1, 1, length, 1]),
                    # TODO: Think about using prediction as a feature for associations.
                ],
                axis=-1,
            )
            associations_flat = tf.layers.dense(
                tf.reshape(features, shape=[-1, nx * 4]), num_associations
            )
            associations = tf.reshape(
                associations_flat, [-1, length, length, num_associations]
            )

            return logits, associations_flat, associations

        with tf.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )

            logits, associations_flat, associations = seq_lab_internal(hidden)

        log_likelihood = 0.0
        association_loss = 0.0
        class_weights = kwargs.get("class_weights")
        if class_weights is not None:
            logits = class_reweighting(class_weights)(logits)

        transition_params = tf.get_variable(
            "Transition_matrix", shape=[n_targets, n_targets]
        )
        if targets is not None:
            log_likelihood, _ = crf_log_likelihood(
                logits,
                targets["labels"],
                kwargs.get("max_length") * tf.ones(tf.shape(targets["labels"])[0]),
                transition_params=transition_params,
            )
            sequence_mask = tf.sequence_mask(
                lengths, maxlen=length, dtype=tf.float32
            )
            mask = tf.expand_dims(sequence_mask, 1) * tf.expand_dims(sequence_mask, 2)

            association_loss = tf.losses.sparse_softmax_cross_entropy(
                logits=associations_flat,
                labels=tf.reshape(targets["associations"], shape=[-1]),
                weights=tf.reshape(mask, shape=[-1]),
            )

        return {
            "logits": {"sequence": logits, "association": associations},
            "losses": -log_likelihood
            + config.assocation_loss_weight
            * association_loss,  # TODO: think about weighting.
            "predict_params": {"transition_matrix": transition_params},
        }

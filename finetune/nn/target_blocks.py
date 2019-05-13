import functools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood

from finetune.base_models.gpt.featurizer import attn, dropout, norm
from finetune.util.shapes import shape_list, merge_leading_dims
from finetune.optimizers.recompute_grads import recompute_grad
from finetune.errors import FinetuneError


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
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b


def language_model(*, X, M, embed_weights, hidden, config, reuse=None):
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
    with tf.variable_scope('model/language-model', reuse=reuse):
        # language model ignores last hidden state because we don't have a target
        sliced_hidden = hidden[:, :-1]
        lm_h = tf.reshape(sliced_hidden, [-1, config.n_embed])  # [batch, seq_len, embed] --> [batch * seq_len, embed]
        lm_logits = tf.matmul(lm_h, embed_weights, transpose_b=True)  # tied weights
        lm_logits = tf.reshape(lm_logits, [batch, seq - 1, tf.shape(embed_weights)[0]])

        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits,
            labels=X[:, 1:, 0]
        )

        perplexity = tf.reduce_sum(tf.exp(lm_losses) * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)


        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])

        # tf.maximum op prevents divide by zero error when mask is all 0s
        lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.maximum(tf.reduce_sum(M[:, 1:], 1), 1)

        lm_logits_shape = shape_list(lm_logits)
        sliced_hidden_shape = shape_list(sliced_hidden)
        return {
            'logits': tf.reshape(lm_logits, shape=sliced_hidden_shape[:-1] + [lm_logits_shape[-1]]),
            'losses': lm_losses,
            'perplexity': perplexity

        }


def _apply_class_weight(losses, targets, class_weights=None):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = tf.reduce_sum(class_weights * tf.to_float(targets), axis=1)
        weights *= tf.to_float(tf.reduce_prod(tf.shape(weights))) / tf.reduce_sum(weights)
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
    with tf.variable_scope('classifier', reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        clf_logits = perceptron(hidden, n_targets, config)
        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=clf_logits,
                labels=tf.stop_gradient(targets)
            )

            clf_losses = _apply_class_weight(clf_losses, targets, kwargs.get('class_weights'))

        return {
            'logits': clf_logits,
            'losses': clf_losses
        }


def multi_choice_question(hidden, targets, n_targets, config, train=False, reuse=None, **kwargs):
    with tf.variable_scope("model", reuse=reuse):
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
                logits=clf_out,
                labels=tf.stop_gradient(targets)
            )

            clf_losses = _apply_class_weight(clf_losses, targets, kwargs.get('class_weights'))

        return {
            'logits': clf_out,
            'losses': clf_losses
        }


def multi_classifier(hidden, targets, n_targets, config, train=False, reuse=None, **kwargs):
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
    with tf.variable_scope('model', reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        clf_logits = perceptron(hidden, n_targets, config)
        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=clf_logits,
                labels=tf.stop_gradient(targets)
            )
            clf_losses = _apply_class_weight(clf_losses, targets, kwargs.get('class_weights'))
        return {
            'logits': clf_logits,
            'losses': clf_losses
        }


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
    with tf.variable_scope('regressor', reuse=reuse):
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
                    "regression_loss needs to be either L1 or L2, instead it is {}".format(config.regression_loss)
                )
        return {
            'logits': outputs,
            'losses': loss
        }

def ordinal_regressor(hidden, targets, n_targets, config, shared_threshold_weights=True, train=False, reuse=None, **kwargs):
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
    with tf.variable_scope('ordinalregressor', reuse=reuse):
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
                logits=logits,
                labels=tf.stop_gradient(targets))
        return {
            'logits': outputs,
            'losses': loss
        }


def class_reweighting(class_weights):
    @tf.custom_gradient
    def custom_grad(logits):
        def grad(g):
            new_g = g * class_weights
            ratio = tf.norm(g) / tf.norm(new_g)
            return new_g * ratio
        return tf.identity(logits), grad
    return custom_grad


def sequence_labeler(hidden, targets, n_targets, config, pad_id, multilabel=False, train=False, reuse=None, **kwargs):
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
    :param dropout_placeholder:
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distribution is not sufficient and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.variable_scope('sequence-labeler', reuse=reuse):
        nx = config.n_embed

        def seq_lab_internal(hidden):
            if config.base_model.is_bidirectional:
                n = hidden
            else:
                attn_fn = functools.partial(attn, scope="seq_label_attn", n_state=nx, n_head=config.seq_num_heads,
                                                resid_pdrop=config.resid_p_drop, attn_pdrop=config.attn_p_drop,
                                                train=train, scale=False, mask=False)
                n = norm(attn_fn(hidden) + hidden, 'seq_label_residual')
            flat_logits = tf.layers.dense(n, n_targets)
            logits = tf.reshape(flat_logits, tf.concat([tf.shape(hidden)[:2], [n_targets]], 0))
            return logits

        with tf.variable_scope('seq_lab_attn'):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(seq_lab_internal, use_entire_scope=True)
            logits = seq_lab_internal(hidden)

        class_weights = kwargs.get('class_weights')
        if class_weights is not None and train:
            class_weights = tf.reshape(class_weights, [1, 1, -1])
            one_hot_class_weights = class_weights * tf.one_hot(targets, depth=n_targets)
            per_token_weights = tf.reduce_sum(one_hot_class_weights, axis=-1, keep_dims=True)
            logits = class_reweighting(per_token_weights)(logits)

        log_likelihood = 0.0

        if multilabel:
            transition_params = []
            logits_individual = tf.unstack(logits, n_targets, axis=-1)
            if targets is not None:
                targets_individual = tf.unstack(targets, n_targets, axis=-1)
            logits = []
            for i in range(n_targets):
                transition_params.append(tf.get_variable("Transition_matrix_{}".format(i), shape=[2, 2]))
                logits.append(tf.stack((logits_individual[pad_id], logits_individual[i]), axis=-1))
                if targets is not None and train and i != pad_id:
                    log_likelihood += crf_log_likelihood(
                        logits[-1],
                        targets_individual[i],
                        kwargs.get('max_length') * tf.ones(tf.shape(targets)[0]),
                        transition_params=transition_params[-1]
                    )[0]
            logits = tf.stack(logits, axis=-1)
        else:
            transition_params = tf.get_variable("Transition_matrix", shape=[n_targets, n_targets])
            if train and targets is not None:
                log_likelihood, _ = crf_log_likelihood(
                    logits,
                    targets,
                    kwargs.get('max_length') * tf.ones(tf.shape(targets)[0]),
                    transition_params=transition_params
                )

        return {
            'logits': logits,
            'losses': -log_likelihood,
            'predict_params': {
                'transition_matrix': transition_params
            }
        }


def association(hidden, pool_idx, targets, n_targets, config, train=False, reuse=None, **kwargs):
    """
    An Attention based sequence labeler model with association.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param pool_idx: the index of the classify tokens along the sequence dimension. [batch_size]
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
    with tf.variable_scope('sequence-labeler', reuse=reuse):
        nx = config.n_embed
        length = config.max_length
        num_associations = len(config.association_types) + 1

        def seq_lab_internal(hidden):
            attn_fn = functools.partial(
                attn, scope="seq_label_attn", n_state=nx, n_head=config.seq_num_heads,
                resid_pdrop=config.resid_p_drop, attn_pdrop=config.attn_p_drop,
                train=train, scale=False, mask=False
            )
            n = norm(attn_fn(hidden) + hidden, 'seq_label_residual')
            flat_logits = tf.layers.dense(n, n_targets)
            logits = tf.reshape(flat_logits, tf.concat([tf.shape(hidden)[:2], [n_targets]], 0))

            association_head = tf.layers.dense(n, nx)
            association_head = tf.reshape(association_head, tf.concat([tf.shape(hidden)[:2], [nx]], 0))

            a = tf.expand_dims(association_head, 1)
            b = tf.expand_dims(association_head, 2)

            features = tf.concat(
                [
                    a - b, a * b,
                    tf.tile(a, [1, length, 1, 1]),
                    tf.tile(b, [1, 1, length, 1]),
                    # TODO: Think about using prediction as a feature for associations.
                ],
                axis=-1
            )
            associations_flat = tf.layers.dense(tf.reshape(features, shape=[-1, nx * 4]), num_associations)
            associations = tf.reshape(associations_flat, [-1, length, length, num_associations])

            return logits, associations_flat, associations

        with tf.variable_scope('seq_lab_attn'):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(seq_lab_internal, use_entire_scope=True)

            logits, associations_flat, associations = seq_lab_internal(hidden)

        log_likelihood = 0.0
        association_loss = 0.0
        class_weights = kwargs.get('class_weights')
        if class_weights is not None:
            logits = class_reweighting(class_weights)(logits)

        transition_params = tf.get_variable("Transition_matrix", shape=[n_targets, n_targets])
        if train and targets is not None:
            log_likelihood, _ = crf_log_likelihood(
                logits,
                targets["labels"],
                kwargs.get('max_length') * tf.ones(tf.shape(targets["labels"])[0]),
                transition_params=transition_params
            )
            sequence_mask = tf.sequence_mask(pool_idx + 1, maxlen=length, dtype=tf.float32)
            mask = tf.expand_dims(sequence_mask, 1) * tf.expand_dims(sequence_mask, 2)

            association_loss = tf.losses.sparse_softmax_cross_entropy(
                logits=associations_flat,
                labels=tf.reshape(targets["associations"], shape=[-1]),
                weights=tf.reshape(mask, shape=[-1])
            )

        return {
            'logits': {"sequence": logits, "association": associations},
            'losses': -log_likelihood + config.assocation_loss_weight * association_loss,  # TODO: think about weighting.
            'predict_params': {
                'transition_matrix': transition_params
            }
        }

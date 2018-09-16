import functools

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood

from finetune.transformer import dropout, embed, block, attn, norm
from finetune.utils import shape_list, merge_leading_dims
from finetune.recompute_grads import recompute_grad


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


def featurizer(X, encoder, dropout_placeholder, config, train=False, reuse=None, max_length=None):
    """
    The transformer element of the finetuning model. Maps from tokens ids to a dense, embedding of the sequence.

    :param X: A tensor of token indexes with shape [batch_size, sequence_length, token_idx]
    :param encoder: A TextEncoder object.
    :param dropout_placeholder: A placeholder, 1 when dropout is on, 0 when it is off.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param max_length: Maximum sequence length.
    :return: A dict containing;
        embed_weights: the word embedding matrix.
        features: The output of the featurizer_final state.
        sequence_features: The output of the featurizer at each timestep.
    """
    initial_shape = [a or -1 for a in X.get_shape().as_list()]
    X = tf.reshape(X, shape=[-1] + initial_shape[-2:])

    max_length = max_length or config.max_length
    with tf.variable_scope('model/featurizer', reuse=reuse):
        embed_weights = tf.get_variable("we", [encoder.vocab_size + max_length, config.n_embed],
                                        initializer=tf.random_normal_initializer(stddev=config.weight_stddev))
        if config.train_embeddings:
            embed_weights = dropout(embed_weights, config.embed_p_drop, train, dropout_placeholder)
        else:
            embed_weights = tf.stop_gradient(embed_weights)

        X = tf.reshape(X, [-1, max_length, 2])

        h = embed(X, embed_weights)
        for layer in range(config.n_layer):
            if (layer - config.n_layer) == config.num_layers_trained and config.num_layers_trained != 12:
                h = tf.stop_gradient(h)
                train_layer = False
            else:
                train_layer = train
            with tf.variable_scope('h%d_' % layer):
                block_fn = functools.partial(block, n_head=config.n_heads, act_fn=config.act_fn,
                                             resid_pdrop=config.resid_p_drop, attn_pdrop=config.attn_p_drop,
                                             scope='h%d' % layer, dropout_placeholder=dropout_placeholder,
                                             train=train_layer, scale=True)
                if config.low_memory_mode and train_layer:
                    block_fn = recompute_grad(block_fn, use_entire_scope=True)
                h = block_fn(h)

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h, [-1, config.n_embed])  # [batch * seq_len, embed]
        clf_token = encoder['_classify_']
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * max_length + pool_idx)
        clf_h = tf.reshape(clf_h, shape=initial_shape[0: -2] + [config.n_embed])
        seq_feats = tf.reshape(h, shape=initial_shape[:-1] + [config.n_embed])

        return {
            'embed_weights': embed_weights,
            'features': clf_h,
            'sequence_features': seq_feats
        }


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

    with tf.variable_scope('model/language-model', reuse=reuse):
        # language model ignores last hidden state because we don't have a target
        lm_h = tf.reshape(hidden[:, :-1], [-1, config.n_embed])  # [batch, seq_len, embed] --> [batch * seq_len, embed]
        lm_logits = tf.matmul(lm_h, embed_weights, transpose_b=True)  # tied weights
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits,
            labels=tf.reshape(X[:, 1:, 0], [-1])
        )

        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
        lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)
        return {
            'logits': lm_logits,
            'losses': lm_losses,
        }


def _apply_class_weight(losses, targets, class_weights=None):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = tf.reduce_sum(class_weights * tf.to_float(targets), axis=1)
        losses *= weights
    return losses


def classifier(hidden, targets, n_targets, dropout_placeholder, config, train=False, reuse=None, **kwargs):
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
        hidden = dropout(hidden, config.clf_p_drop, train, dropout_placeholder)
        clf_logits = perceptron(hidden, n_targets, config)
        clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=clf_logits,
            labels=tf.stop_gradient(targets)
        )

        clf_losses = _apply_class_weight(clf_losses, targets, kwargs.get('class_weights'))

        return {
            'logits': clf_logits,
            'losses': clf_losses
        }


def multi_choice_question(hidden, targets, n_targets, dropout_placeholder, config, train=False, reuse=None, **kwargs):
    with tf.variable_scope("model", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train, dropout_placeholder)
        hidden = tf.unstack(hidden, num=n_targets, axis=1)
        hidden = tf.concat(hidden, axis=0)
        # some model
        clf_out = perceptron(hidden, 1, config)
        clf_out = tf.split(clf_out, n_targets, axis=0)
        clf_out = tf.concat(clf_out, 1)

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=clf_out,
            labels=tf.stop_gradient(targets)
        )

        clf_losses = _apply_class_weight(clf_losses, targets, kwargs.get('class_weights'))

        return {
            'logits': clf_out,
            'losses': clf_losses
        }


def multi_classifier(hidden, targets, n_targets, dropout_placeholder, config, train=False, reuse=None, **kwargs):
    """
    A simple linear classifier.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the sparse targets [batch_size, n_targets]
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
    with tf.variable_scope('model', reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train, dropout_placeholder)
        clf_logits = perceptron(hidden, n_targets, config)
        clf_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=clf_logits,
            labels=tf.stop_gradient(targets)
        )
        clf_losses = _apply_class_weight(clf_losses, targets, kwargs.get('class_weights'))
        return {
            'logits': clf_logits,
            'losses': clf_losses
        }


def regressor(hidden, targets, n_targets, dropout_placeholder, config, train=False, reuse=None, **kwargs):
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
        hidden = dropout(hidden, config.clf_p_drop, train, dropout_placeholder)
        outputs = perceptron(hidden, n_targets, config)
        loss = tf.nn.l2_loss(outputs - targets)
        return {
            'logits': outputs,
            'losses': loss
        }


def class_reweighting(class_weights):
    @tf.custom_gradient
    def custom_grad(logits):
        def grad(g):
            return g * class_weights
        return tf.identity(logits), grad
    return custom_grad


def sequence_labeler(hidden, targets, n_targets, dropout_placeholder, config, train=False, reuse=None, **kwargs):
    """
    An Attention based sequence labeler model. Takes the output of the pre-trained model, applies an additional
    randomly initialised multihead attention block, with residuals on top. The attention is not-future masked to allow
    the model to label sequences based on context in both directions. The representations fed into this model are
    necessarily future masked because a language modelling loss is the original objective of the featurizer.

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
            sampling from this distrobution is not sufficiant and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.variable_scope('sequence-labeler', reuse=reuse):
        nx = config.n_embed
        def seq_lab_internal(hidden):
            attn_fn = functools.partial(attn, scope="seq_label_attn", n_state=nx, n_head=config.seq_num_heads,
                                            resid_pdrop=config.resid_p_drop, attn_pdrop=config.attn_p_drop,
                                            dropout_placeholder=dropout_placeholder, train=train, scale=False, mask=False)
            n = norm(attn_fn(hidden) + hidden, 'seq_label_residual')
            flat_logits = tf.layers.dense(n, n_targets)
            logits = tf.reshape(flat_logits, tf.concat([tf.shape(hidden)[:2], [n_targets]], 0))
            return logits

        with tf.variable_scope('seq_lab_attn'):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(seq_lab_internal, use_entire_scope=True)
            logits = seq_lab_internal(hidden)

        transition_params = tf.get_variable("Transition_matrix", shape=[n_targets, n_targets])

        class_weights = kwargs.get('class_weights')
        if class_weights is not None:
            logits = class_reweighting(class_weights)(logits)
        
        if train:
            log_likelihood, _ = crf_log_likelihood(
                logits, 
                targets,
                kwargs.get('max_length') * tf.ones(tf.shape(targets)[0]),
                transition_params=transition_params
            )
        else:
            log_likelihood = tf.constant(0.)

        return {
            'logits': logits,
            'losses': -log_likelihood,
            'predict_params': {
                'transition_matrix': transition_params
            }
        }


def cosine_similarity(hidden1, hidden2, targets, n_targets, dropout_placeholder, config, train=False, reuse=None, **kwargs):
    """
    A simple model to compute cosine similarity between two document embeddings.

    :param hidden1: The output of the featurizer for document 1. [batch_size, embed_dim]
    :param hidden2: The output of the featurizer for document 2. [batch_size, embed_dim]
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
    with tf.variable_scope('cosine_similarity', reuse=reuse):
        hidden1 = dropout(hidden1, config.clf_p_drop, train, dropout_placeholder)
        hidden2 = dropout(hidden2, config.clf_p_drop, train, dropout_placeholder)
        cos_sim_logits = tf.sigmoid(tf.multiply(hidden1, hidden2))

        cos_sim_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=cos_sim_logits,
            labels=tf.stop_gradient(targets)
        )

        cos_sim_losses = _apply_class_weight(cos_sim_losses, targets, kwargs.get('class_weights'))

        return {
            'logits': cos_sim_logits,
            'losses': cos_sim_losses
        }

from finetune.transformer import dropout, embed, block, attn, norm
from finetune.utils import shape_list
import tensorflow as tf

def merge_leading_dims(X, target_rank):
    shape = [-1] + X.get_shape().as_list()[1 - target_rank:]
    return tf.reshape(X, shape)

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
    with tf.variable_scope('clf'):
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
    with tf.variable_scope('model', reuse=reuse):
        embed_weights = tf.get_variable("we", [encoder.vocab_size + max_length, config.n_embed],
                                        initializer=tf.random_normal_initializer(stddev=config.weight_stddev))
        embed_weights = dropout(embed_weights, config.embed_p_drop, train, dropout_placeholder)

        X = tf.reshape(X, [-1, max_length, 2])

        h = embed(X, embed_weights)
        for layer in range(config.n_layer):
            h = block(h, config.n_heads, config.act_fn, config.resid_p_drop, config.attn_p_drop, 'h%d' % layer,
                      dropout_placeholder, train=train, scale=True)
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

    with tf.variable_scope('model', reuse=reuse):
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


def classifier(hidden, targets, n_targets, dropout_placeholder, config, train=False, reuse=None, **kwargs):
    """
    A simple linear classifier.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the sparse target ids. [batch_size]
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
        clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=clf_logits,
            labels=tf.stop_gradient(targets)
        )
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
    with tf.variable_scope('model', reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train, dropout_placeholder)
        outputs = perceptron(hidden, n_targets, config)
        loss = tf.nn.l2_loss(outputs - targets)
        return {
            'logits': outputs,
            'losses': loss
        }


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
    with tf.variable_scope('model/clf', reuse=reuse):
        nx = config.n_embed
        a = attn(hidden, 'seq_label_attn', nx, config.seq_num_heads, config.seq_dropout, config.seq_dropout, dropout_placeholder, train=train, scale=False, mask=False)
        n = norm(hidden + a, 'seq_label_residual')
        flat_logits = tf.layers.dense(n, n_targets)
        logits = tf.reshape(flat_logits, tf.concat([tf.shape(hidden)[:2], [n_targets]], 0))
        # TODO (BEN): ADD: correct way to find lengths. - Same method in decoding. Cheating for now.
        with tf.device(None):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, targets, kwargs.get('max_length') * tf.ones(tf.shape(targets)[0]))
        return {
            'logits': logits,
            'losses': -log_likelihood,
            'predict_params': {
                'transition_matrix': transition_params
            }
        }

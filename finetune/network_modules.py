from finetune.config import WEIGHT_STDDEV, MAX_LENGTH, EMBED_P_DROP, N_LAYER, N_HEADS, ACT_FN, RESID_P_DROP, \
    ATTN_P_DROP, CLF_P_DROP, N_EMBED
from finetune.transformer import dropout, embed, block
from finetune.utils import shape_list
import tensorflow as tf


def mlp(x, ny, w_init=tf.random_normal_initializer(stddev=WEIGHT_STDDEV), b_init=tf.constant_initializer(0)):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b


def featurizer(X, encoder, dropout_placeholder, train=False, reuse=None, max_length=MAX_LENGTH):
    with tf.variable_scope('model', reuse=reuse):
        embed_weights = tf.get_variable("we", [encoder.vocab_size + max_length, N_EMBED],
                                        initializer=tf.random_normal_initializer(stddev=WEIGHT_STDDEV))
        embed_weights = dropout(embed_weights, EMBED_P_DROP, train, dropout_placeholder)

        X = tf.reshape(X, [-1, max_length, 2])

        h = embed(X, embed_weights)
        for layer in range(N_LAYER):
            h = block(h, N_HEADS, ACT_FN, RESID_P_DROP, ATTN_P_DROP, 'h%d' % layer, dropout_placeholder, train=train, scale=True)

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h, [-1, N_EMBED])  # [batch * seq_len, embed]
        clf_token = encoder['_classify_']
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * max_length + pool_idx)

        clf_h = tf.reshape(clf_h, [-1, N_EMBED])  # [batch, embed]
        return {
            'embed_weights': embed_weights,
            'features': clf_h,
            'sequence_features': h
        }


def language_model(*, X, M, embed_weights, hidden, reuse=None):
    with tf.variable_scope('model', reuse=reuse):
        # language model ignores last hidden state because we don't have a target
        lm_h = tf.reshape(hidden[:, :-1], [-1, N_EMBED])  # [batch, seq_len, embed] --> [batch * seq_len, embed]
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


def classifier(hidden, targets, n_classes, dropout_placeholder, train=False, reuse=None):
    with tf.variable_scope('model', reuse=reuse):
        hidden = dropout(hidden, CLF_P_DROP, train, dropout_placeholder)
        clf_logits = mlp(hidden, n_classes)
        clf_losses = tf.nn.softmax_cross_entropy_with_logits(logits=clf_logits, labels=targets)
        return {
            'logits': clf_logits,
            'losses': clf_losses
        }


def regressor(hidden, targets, n_outputs, dropout_placeholder, train=False, reuse=None):
    with tf.variable_scope('model', reuse=reuse):
        hidden = dropout(hidden, CLF_P_DROP, train, dropout_placeholder)
        outputs = mlp(hidden, n_outputs)
        loss = tf.nn.l2_loss(outputs - targets)
        return {
            'logits': outputs,
            'losses': loss
        }

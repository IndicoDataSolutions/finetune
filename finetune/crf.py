import numpy as np
import tensorflow as tf

from finetune.utils import np_softmax


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag
            indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    return viterbi, np_softmax(trellis, axis=-1)


def sequence_decode(logits, transition_matrix):
    """ A simple py_func wrapper around the Viterbi decode allowing it to be included in the tensorflow graph. """

    def _sequence_decode(logits, transition_matrix):
        all_predictions = []
        all_logits = []
        for logit in logits:
            viterbi_sequence, viterbi_logits = viterbi_decode(logit, transition_matrix)
            all_predictions.append(viterbi_sequence)
            all_logits.append(viterbi_logits)
        return np.array(all_predictions, dtype=np.int32), np.array(all_logits, dtype=np.float32)

    return tf.py_func(_sequence_decode, [logits, transition_matrix], [tf.int32, tf.float32])

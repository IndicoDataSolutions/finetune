import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def np_softmax(x, t=1, axis=-1):
    x = x / t
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


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

def top_k_0_axis(matrix, k):
    matrix = tf.transpose(matrix)
    scores, indicies = tf.math.top_k(matrix, k=k)
    scores = tf.transpose(scores)
    indicies = tf.transpose(indicies)
    return scores, indicies

def k_viterbi_decode(tag_sequence, transition_matrix, top_k=5):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    top_k : int, required.
        Integer defining the top number of paths to decode.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : float
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.shape())

    path_scores = []
    path_indices = []
    # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
    # to allow for 1 permutation.
    path_scores.append(tf.expand_dims(tag_sequence[0, :], 0))
    # assert path_scores[0].size() == (n_permutations, num_tags)

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
        summed_potentials = tf.expand_dims(path_scores[timestep - 1], 2) \
                + transition_matrix
        summed_potentials = tf.reshape(summed_potentials, (-1, num_tags))

        # Best pairwise potential path score from the previous timestep. 
        max_k = min(summed_potentials.shape()[0], top_k)
        scores, paths = top_k_0_axis(summed_potentials, max_k)
        # assert scores.size() == (n_permutations, num_tags)
        # assert paths.size() == (n_permutations, num_tags)

        scores = tag_sequence[timestep, :] + scores
        # assert scores.size() == (n_permutations, num_tags)
        path_scores.append(scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores = tf.reshape(path_scores[-1], (-1,))
    max_k = min(path_scores.shape()[0], top_k)

    viterbi_scores, best_paths = top_k_0_axis(path_scores, max_k)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(tf.reshape(backward_timestep, (-1,))[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    return viterbi_paths, viterbi_scores

def k_best_sequence_decode(logits, transition_matrix, k):
    def _sequence_decode(logits, transition_matrix):
        all_predictions = []
        all_logits = []
        for logit in logits:
            viterbi_sequence, viterbi_logits = k_viterbi_decode(logit,
                                                                transition_matrix,
                                                                top_k=k)
            all_predictions.append(viterbi_sequence)
            all_logits.append(viterbi_logits)
        return np.array(all_predictions, dtype=np.int32), np.array(all_logits, dtype=np.float32)
    return tf.compat.v1.py_func(_sequence_decode, [logits, transition_matrix], [tf.int32, tf.float32])

def sequence_decode(logits, transition_matrix, sequence_length, use_gpu_op, use_crf):
    """ A simple py_func wrapper around the Viterbi decode allowing it to be included in the tensorflow graph. """
    if not use_crf:
        return tf.argmax(input=logits, axis=-1), tf.nn.softmax(logits, -1)

    if use_gpu_op:
        tags, _ = tfa.text.crf.crf_decode(
            logits,
            transition_matrix,
            sequence_length
        )
        probs = tf.nn.softmax(logits, -1)
        return tags, probs
    else:
        def _sequence_decode(logits, transition_matrix):
            all_predictions = []
            all_logits = []
            for logit in logits:
                viterbi_sequence, viterbi_logits = viterbi_decode(logit, transition_matrix)
                all_predictions.append(viterbi_sequence)
                all_logits.append(viterbi_logits)
            return np.array(all_predictions, dtype=np.int32), np.array(all_logits, dtype=np.float32)
        
        return tf.compat.v1.py_func(_sequence_decode, [logits, transition_matrix], [tf.int32, tf.float32])

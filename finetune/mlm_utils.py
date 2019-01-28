import random
import tensorflow as tf


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_len):
    """ Re-purposed from Bert Repo """
    cand_indexes = list(range(len(tokens)))
    random.shuffle(cand_indexes)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    cand_indexes = cand_indexes[:num_to_predict]

    output_tokens = list(tokens)

    to_predict = []

    for index, curr_tok in enumerate(tokens):
        if index not in cand_indexes:
            to_predict.append(0.0)
            continue
        to_predict.append(1.0)

        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            to_predict.append(1.0)
            masked_token = vocab_len  # mask token is the final token.
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = curr_tok
            # 10% of the time, replace with random word
            else:
                masked_token = random.randint(0, vocab_len - 1)

        output_tokens[index] = masked_token

    return output_tokens, to_predict


def get_bert_process_op(features, max_predictions_per_seq, vocab_len, masked_lm_prob):
    def tf_bert_process(token):
        return tf.py_func(
            lambda _token: create_masked_lm_predictions(_token, masked_lm_prob, max_predictions_per_seq, vocab_len),
            [token],
            (tf.int32, tf.float32),
            stateful=False,
        )

    output_features, to_predict = tf.map_fn(
        tf_bert_process,
        features,
        dtype=(tf.int32, tf.float32)
    )
    return output_features, to_predict
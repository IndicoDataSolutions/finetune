import tensorflow as tf

from finetune.util.shapes import shape_list


def sample_with_temperature(logits, temperature):
    """Either argmax or random sampling.
    Args:
      logits: a Tensor.
      temperature: a float  0.0=argmax 1.0=random
    Returns:
      a Tensor with one fewer dimension than logits.
    """
    logits_shape = shape_list(logits)
    if temperature == 0.0:
        return tf.argmax(input=logits, axis=-1)
    else:
        assert temperature > 0.0
        reshaped_logits = tf.reshape(logits, [-1, logits_shape[-1]]) / temperature
        choices = tf.random.categorical(logits=reshaped_logits, num_samples=1)
        choices = tf.reshape(choices, logits_shape[:-1])
        return choices

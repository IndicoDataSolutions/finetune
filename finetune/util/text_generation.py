import tensorflow as tf

from finetune.util.shapes import shape_list


def format_mlm_predictions(masked_tensor, gen_text_key, mlm_ids_key, mlm_positions_key, encoder):
    
    # Get the number of masks to predict from one of the predictions
    
    number_of_masks = gen_text_key.shape[0]
    k=5
    #return ' '.join([
    #    str(gen_text_key.shape[0]),
    #    str(mlm_ids_key.shape[0]),
    #    str(mlm_positions_key.shape[0])])
    
    predicted_tokens = [{'prediction_ids': [encoder.decode([i]) for i in gen_text_key[mask][-k:][::-1]],
                         'original_token_id': encoder.decode([mlm_ids_key[mask]]),
                         'position': mlm_positions_key[mask]} for mask in range(number_of_masks)]

    mask_positions = [i['position'] for i in predicted_tokens]

    #tokens = encoder._encode([input_text]).tokens[0]
    tokens = [encoder.decode([i]) for i in masked_tensor[0][:, 0]]

    mask_number = iter(range(len(predicted_tokens)))
    text_to_display = "".join([tokens[i] if i+1 not in mask_positions
                                else '<' + str(next(mask_number)) + '>'
                                for i in range(len(tokens))]) + 2*'\n'

    for i in sorted(predicted_tokens, key=lambda x: x['position']):
        text_to_display += (f"{'<':>3}{mask_positions.index(i['position']):>2}>|{i['original_token_id']:15}|{i['prediction_ids']}\n")

    return text_to_display

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
        return tf.argmax(logits, axis=-1)
    else:
        assert temperature > 0.0
        reshaped_logits = tf.reshape(logits, [-1, logits_shape[-1]]) / temperature
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices, logits_shape[:-1])
        return choices

import numpy as np
from scipy import interpolate


def interpolate_pos_embed(positional_embed, new_len):
    xx = np.linspace(0, 512, new_len)
    newKernel = interpolate.RectBivariateSpline(np.arange(positional_embed.shape[0]),
                                                np.arange(positional_embed.shape[1]), positional_embed)
    return newKernel(xx, np.arange(positional_embed.shape[1]))


def process_pos_embed(positional_embed, max_length, interpolate):
    if interpolate and max_length != len(positional_embed):
        positional_embed = interpolate_pos_embed(positional_embed, max_length)
        
    elif max_length > len(positional_embed):
        raise ValueError("Max Length cannot be greater than {} if interpolate_pos_embed is turned off".format(
            len(positional_embed)))
    else:
        positional_embed = positional_embed[:max_length]
    return positional_embed
                         

def embedding_preprocessor(input_pipeline, config):

    def process_embeddings(name, value):
        if "/we:0" in name:
            vocab_size = input_pipeline.text_encoder.vocab_size
            word_embeddings = value[:vocab_size - len(input_pipeline.text_encoder.special_tokens)]
            special_embed = value[len(word_embeddings): vocab_size]
            positional_embed = value[vocab_size:]

            positional_embed = process_pos_embed(positional_embed, config.max_length, config.interpolate_pos_embed)
            value = np.concatenate((word_embeddings, special_embed, positional_embed), axis=0)
            
        elif "position_embeddings" in name:
            value = process_pos_embed(value, config.max_length, config.interpolate_pos_embed)
            
        return value
        
        
    return process_embeddings

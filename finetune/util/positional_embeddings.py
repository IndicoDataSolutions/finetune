import numpy as np
from scipy import interpolate


def interpolate_pos_embed(positional_embed, new_len):
    xx = np.linspace(0, 512, new_len)
    newKernel = interpolate.RectBivariateSpline(np.arange(positional_embed.shape[0]),
                                                np.arange(positional_embed.shape[1]), positional_embed)
    return newKernel(xx, np.arange(positional_embed.shape[1]))


def embedding_preprocessor(input_pipeline, config):

    def process_embeddings(name, value):
        if "/we:0" not in name:
            return value

        vocab_size = input_pipeline.text_encoder.vocab_size
        word_embeddings = value[:vocab_size]
        positional_embed = value[vocab_size:]

        if config.interpolate_pos_embed and config.max_length != len(positional_embed):
            positional_embed = interpolate_pos_embed(positional_embed, config.max_length)

        elif config.max_length > len(positional_embed):
            raise ValueError("Max Length cannot be greater than {} if interpolate_pos_embed is turned off".format(
                len(positional_embed)))
        else:
            positional_embed = positional_embed[:config.max_length]

        embeddings = np.concatenate((word_embeddings, positional_embed), axis=0)
        return embeddings
        
    return process_embeddings

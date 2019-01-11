import numpy as np
from scipy import interpolate


def interpolate_pos_embed(positional_embed, new_len):
    xx = np.linspace(0, 512, new_len)
    newKernel = interpolate.RectBivariateSpline(np.arange(positional_embed.shape[0]),
                                                np.arange(positional_embed.shape[1]), positional_embed)
    return newKernel(xx, np.arange(positional_embed.shape[1]))

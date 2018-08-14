import tensorflow as tf
from tensorflow.python.client import device_lib
from functools import lru_cache
from collections import namedtuple

# CONSTANTS
PAD_TOKEN = '<PAD>'


@lru_cache()
def all_gpus():
    """
    Get integer ids of all available GPUs
    """
    local_device_protos = device_lib.list_local_devices()
    return [
        int(x.name.split(':')[-1]) for x in local_device_protos
        if x.device_type == 'GPU'
    ]

GridSearchable = namedtuple("GridSearchable", "default iterator")

class Settings(dict):

    def get_grid_searchable(self):
        return self.grid_searchable

    def __init__(self, **kwargs):
        super().__init__()
        self.grid_searchable = {}
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    def __setitem__(self, key, value):
        if isinstance(value, GridSearchable):
            self.grid_searchable[key] = value.iterator
            value = value.default
        return super().__setitem__(key, value)

    def __setattr__(self, k, v):
        return self.__setitem__(k, v)

    __delattr__ = dict.__delitem__


def get_default_config():
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :return: Config object.
    """
    return Settings(
        # MODEL DEFINITION (DO NOT CHANGE)
        n_heads=12,
        n_layer=12,
        act_fn="gelu",
        n_embed=768,

        # TRAINING SETTINGS
        batch_size=2,
        visible_gpus=all_gpus(),
        n_epochs=GridSearchable(3, [1, 2, 3, 4]),
        seed=42,
        max_length=512,
        # INITIALIZATION
        weight_stddev=0.02,

        # LONG SEQUENCE
        low_memory_mode=False,
        interpolate_pos_embed=True,

        # REGULARIZATION
        embed_p_drop=0.1,
        attn_p_drop=0.1,
        resid_p_drop=0.1,
        clf_p_drop=0.1,
        l2_reg=GridSearchable(0.0, [0.0, 0.1, 0.2]),
        vector_l2=True,

        # LOSS + OPTIMIZATION
        b1=0.9,
        b2=0.999,
        epsilon=1e-8,
        lr_schedule='warmup_linear',
        lr=GridSearchable(6.25e-5, [6.25e-4, 6.25e-5, 6.25e-6]),
        lr_warmup=0.002,
        max_grad_norm=1,
        lm_loss_coef=0.0,
        rolling_avg_decay=0.99,
        regularize_deviation=0.0,

        # Logging
        summarize_grads=False,
        verbose=True,

        # Validation
        val_size=0.05,
        val_interval=150,
        val_window_size=5,

        # Language Modelling output.
        lm_temp=0.2,

        # Sequence Labeling
        seq_num_heads=16,
        seq_dropout=0.3,
        pad_token="<PAD>",
        subtoken_predictions=False,

        # Multilabel
        multi_label_threshold=0.5,

        # Early stopping
        save_best_model=False,
        autosave_path=None,

        # Tensorboard
        tensorboard_folder=None,

        # debugging
        log_device_placement=False,
        soft_device_placement=True,

        # Save options
        save_adam_vars=True,
       
        # Dealing with long sequences
        chunk_long_sequences=False,
    )


def get_config(**kwargs):
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :param **kwargs: Keyword arguments to override default values.
    :return: Config object.    """
    config = get_default_config()
    config.update(kwargs)
    return config


def cpu_config():
    config = get_default_config()
    config.visible_gpus = []
    return config

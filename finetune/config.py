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
        # MODEL DEFINITION 
        # ----------------
        # Number of heads for the multihead attention block of transformer
        n_heads=12, 

        # Number of transformer layers
        n_layer=12,

        # Activation function
        act_fn="gelu",

        # Embedding size
        n_embed=768,


        # TRAINING SETTINGS
        # -----------------
        # Number of examples per batch
        batch_size=2,

        # List of integer GPU ids to spread out computation across
        visible_gpus=all_gpus(),

        # Number of iterations through training data
        n_epochs=GridSearchable(3, [1, 2, 3, 4]),

        # Random seed to use for repeatability purposes
        seed=42,

        # Maximum number of subtokens per sequence.
        # Examples longer than this number will be truncated
        # (unless chunk_long_sequences=True for SequenceLabeler models)
        max_length=512,


        # INITIALIZATION
        # --------------
        # Standard deviation of initial weights
        weight_stddev=0.02,


        # LONG SEQUENCES
        # --------------
        # When True, use a sliding window approach to predict on 
        # examples that are longer than max length
        chunk_long_sequences=False,

        # When True, only store partial gradients on forward pass
        # and recompute remaining gradients incrementally in order 
        # to save memory
        low_memory_mode=False,

        # Interpolate positional embeddings when max_length differs
        # from it's original value of 512
        interpolate_pos_embed=True,


        # REGULARIZATION
        # --------------
        # Embedding dropout probability
        embed_p_drop=0.1,

        # Attention dropout probability
        attn_p_drop=0.1,

        # Residual layer fully connected network dropout probability
        resid_p_drop=0.1,

        # Classifier dropout probability
        clf_p_drop=0.1,

        # L2 regularization coefficient
        l2_reg=GridSearchable(0.0, [0.0, 0.1, 0.2]),

        # Should the l2 penalty be applied to biases?
        vector_l2=True,

        # L2 penalty against pre-trained model weights
        regularize_deviation=0.0,


        # LOSS + OPTIMIZATION
        # -------------------
        # Adam b1 parameter
        b1=0.9,

        # Adam b2 parameter
        b2=0.999,

        # Adam epsilon parameter
        epsilon=1e-8,

        # Learning rate schedule -- see finetune/optimizers.py for more options
        lr_schedule='warmup_linear',

        # Learning rate
        lr=GridSearchable(6.25e-5, [6.25e-4, 6.25e-5, 6.25e-6]),

        # Learning rate warmup (percentage of all batches to warmup for)
        lr_warmup=0.002,

        # Clip gradients larger than this norm
        max_grad_norm=1,

        # Language modeling loss coefficient -- a value between 0 - 1
        # that indicates how to trade off between language modeling loss
        # and target model loss.  Usually not beneficial to turn on unless 
        # dataset size exceeds a few thousand examples.
        lm_loss_coef=0.0,


        # LOGGING
        # -------
        # Include gradient summary information in tensorboard 
        summarize_grads=False,
        
        # Print TQDM logs
        verbose=True,


        # VALIDATION
        # ----------
        # Validation set size as a percentage of all training data
        val_size=0.05,
        
        # Evaluate on validation set after `val_interval` batches
        val_interval=150,

        # Print running average of validation score over
        # `val_window_size` batches
        val_window_size=5,

        # Momentum-style parameter to smooth out validation estimates
        # printed during training
        rolling_avg_decay=0.99


        # TEXT GENERATION
        # ---------------
        # Language model temperature -- a value of 0. corresponds to greedy
        # maximum likelihood predictions while a value of 1. corresponds to 
        # random predictions
        lm_temp=0.2,


        # SEQUENCE LABELING
        # -----------------
        # Number of attention heads of final attention layer
        seq_num_heads=16,

        # Label for unlabeled tokens
        pad_token="<PAD>",

        # Return predictions at subtoken granularity or token granularity?
        subtoken_predictions=False,


        # MULTILABEL
        # ----------
        # Threshold of sigmoid unit in multi label classifier.
        # Can be increased or lowered to trade off precision / recall.
        multi_label_threshold=0.5,


        # EARLY STOPPING
        # --------------
        # Save current best model (as measured by validation loss) to this location 
        autosave_path=None,


        # TENSORBOARD
        # -----------
        # Directory for tensorboard logs.
        # Tensorboard logs will not be written unless tensorboard_folder is explicitly provided
        tensorboard_folder=None,


        # DEBUGGING
        # ----------
        # Log which device each operation is placed on for debugging purposes
        log_device_placement=False,
        # Allow tf to allocate an operation to a different device if a device is unavailable
        soft_device_placement=True,


        # SAVE OPTIONS
        # ------------
        # Save adam parameters when calling model.save()
        save_adam_vars=True,       
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

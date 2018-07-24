import tensorflow as tf

# CONSTANTS
PAD_TOKEN = '<PAD>'


def get_default_config():
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :return: Config object.
    """
    return tf.contrib.training.HParams(
        # MODEL DEFINITION (DO NOT CHANGE)
        n_heads=12,
        n_layer=12,
        act_fn="gelu",
        n_embed=768,

        # TRAINING SETTINGS
        batch_size=2,
        visible_gpus=None,
        n_epochs=3,
        seed=42,
        max_length=256,

        # INITIALIZATION
        weight_stddev=0.02,

        # REGULARIZATION
        embed_p_drop=0.1,
        attn_p_drop=0.1,
        resid_p_drop=0.1,
        clf_p_drop=0.1,
        l2_reg=0.01,
        vector_l2=True,

        # LOSS + OPTIMIZATION
        b1=0.9,
        b2=0.999,
        epsilon=1e-8,
        lr_schedule='warmup_linear',
        lr=6.25e-5,
        lr_warmup=0.002,
        max_grad_norm=1,
        lm_loss_coef=0.5,
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

        # Early stopping
        save_best_model=False,
        autosave_path=None,

        # Tensorboard
        tensorboard_folder=None
    )


def get_config(**kwargs):
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :param **kwargs: Keyword arguments to override default values.
    :return: Config object.    """
    config = get_default_config()
    config.override_from_dict(kwargs)
    return config


def cpu_config():
    config = get_default_config()
    config.visibleGpus = []
    return config

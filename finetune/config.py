import tensorflow as tf


def get_default_hparams():
    return tf.contrib.training.HParams(
        # TRAINING SETTINGS
        batch_size=2,
        visible_gpus=None,
        n_epochs=3,
        seed=42,

        # MODEL DEFINITION + INITIALIZATION
        weight_stddev=0.02,
        max_length=256,
        n_heads=12,
        n_layer=12,
        act_fn="gelu",
        n_embed=768,

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

        # Logging
        summarize_grads=False,

        # Validation
        val_size=0.05,
        val_interval=150,
        val_window_size=5,

        # Language Modelling output.
        lm_temp=0.2

    )


def get_hparams(**kwargs):
    hparams = get_default_hparams()
    for k, v in kwargs.items():
        setattr(hparams, k, v)
    return hparams


def cpu_hparams():
    hparam = get_default_hparams()
    hparam.visibleGpus = []
    return hparam

import tensorflow as tf


def get_default_hparms():
    return tf.contrib.training.HParams(
        # TRAINING SETTINGS
        batchSize=4,
        visibleGpus=None,
        nEpochs=3,
        seed=42,

        # MODEL DEFINITION + INITIALIZATION
        weightStddev=0.02,
        maxLength=512,
        nHeads=12,
        nLayer=12,
        actFn="gelu",
        nEmbed=768,

        # REGULARIZATION
        embedPDrop=0.1,
        attnPDrop=0.1,
        residPDrop=0.1,
        clfPDrop=0.1,
        l2Reg=0.01,
        vectorL2=True,

        # LOSS + OPTIMIZATION
        B1=0.9,
        B2=0.999,
        epsilon=1e-8,
        lrSchedule='warmup_linear',
        lr=6.25e-5,
        lrWarmup=0.002,
        maxGradNorm=1,
        lmLossCoef=0.5,
        rollingAvgDecay=0.99,

        # Logging
        summarizeGrads=False,

        # Validation
        # Validation
        val_size=0.05,
        val_interval=150,
        val_window_size=5
    )


def cpu_hparams():
    hparam = get_default_hparms()
    hparam.visibleGpus = []
    return hparam

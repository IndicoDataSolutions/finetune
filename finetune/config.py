import logging
import os
import os.path
import subprocess
import warnings
from collections import namedtuple
from functools import lru_cache

import numpy as np
from nltk.metrics.distance import edit_distance

import finetune
from finetune.errors import FinetuneError
from finetune.base_models import GPTModel, GPT2Model, BERT

LOGGER = logging.getLogger('finetune')


def finetune_model_path(path):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(finetune.__file__),
            'model',
            path
        )
    )


@lru_cache()
def all_gpus(visible_gpus=None):
    """
    Get integer ids of all available GPUs.

    Sample response from nvidia-smi -L:
        GPU 0: GeForce GTX 980 (UUID: GPU-2d683060-957f-d5ad-123c-a5b49b0116d9)
        GPU 1: GeForce GTX 980 (UUID: GPU-7b8496dc-3eaf-8db7-01e7-c4a884f66acf)
        GPU 2: GeForce GTX TITAN X (UUID: GPU-9e01f108-e7de-becd-2589-966dcc1c778f)
    """
    if visible_gpus is not None:
        visible_gpus = [int(gpu) for gpu in visible_gpus]
    try:
        sp = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        response = sp.communicate()[0]
        gpu_list = response.decode('utf-8').strip().split('\n')
        device_ids = {}
        for i, gpu in enumerate(gpu_list):
            # May be worth logging GPU description
            device_id_str, _, description = gpu.partition(':')
            assert int(device_id_str.split(' ')[-1]) == i
            device_ids[i] = description

        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")

        # restricting GPUs based on env vars
        if cuda_visible_devices:
            device_ids = {
                device_id: description
                for device_id, description in device_ids.items()
                if str(device_id) in cuda_visible_devices.split(',')
            }

        # restricting GPUs based on config
        if visible_gpus is not None:
            device_ids = {
                device_id: description
                for device_id, description in device_ids.items()
                if device_id in visible_gpus
            }

        LOGGER.info(" Visible GPUs: {{{}}}".format(
            ", ".join([
                "{}:{}".format(device_id, description.split('(')[0]).strip()
                for device_id, description in device_ids.items()
            ])
        ))

        device_ids = list(device_ids.keys())
    except:
        # Failed to parse out available GPUs properly
        warnings.warn("Failed to find available GPUS.  Falling back to CPU only mode.")
        device_ids = []

    return device_ids

GridSearchable = namedtuple("GridSearchable", "default iterator")


class Settings(dict):
    """
    Model configuration options

    :param batch_size: Number of examples per batch, defaults to `2`.
    :param visible_gpus: List of integer GPU ids to spread out computation across, defaults to all available GPUs.
    :param n_epochs: Number of iterations through training data, defaults to `3`.
    :param random_seed: Random seed to use for repeatability purposes, defaults to `42`.
    :param max_length:  Maximum number of subtokens per sequence. Examples longer than this number will be truncated
        (unless `chunk_long_sequences=True` for SequenceLabeler models). Defaults to `512`.
    :param weight_stddev: Standard deviation of initial weights.  Defaults to `0.02`.
    :param chunk_long_sequences: When True, use a sliding window approach to predict on
        examples that are longer than max length.  Defaults to `False`.
    :param low_memory_mode: When True, only store partial gradients on forward pass
        and recompute remaining gradients incrementally in order to save memory.  Defaults to `False`.
    :param interpolate_pos_embed: Interpolate positional embeddings when `max_length` differs from it's original value of
        `512`. Defaults to `False`.
    :param embed_p_drop: Embedding dropout probability.  Defaults to `0.1`.
    :param attn_p_drop: Attention dropout probability.  Defaults to `0.1`.
    :param resid_p_drop: Residual layer fully connected network dropout probability.  Defaults to `0.1`.
    :param clf_p_drop: Classifier dropout probability.  Defaults to `0.1`.
    :param l2_reg: L2 regularization coefficient. Defaults to `0.01`.
    :param vector_l2: Whether to apply weight decay regularization to vectors (biases, normalization etc..). Defaults to False.
    :param optimizer: Optimizer to use, current options include AdamW or AdamaxW.
    :param b1: Adam b1 parameter.  Defaults to `0.9`.
    :param b2: Adam b2 parameter.  Defaults to `0.999`.
    :param epsilon: Adam epsilon parameter: Defaults to `1e-8`.
    :param lr_schedule: Learning rate schedule -- see `finetune/optimizers.py` for more options.
    :param lr: Learning rate.  Defaults to `6.25e-5`.
    :param lr_warmup: Learning rate warmup (percentage of all batches to warmup for).  Defaults to `0.002`.
    :param max_grad_norm: Clip gradients larger than this norm. Defaults to `1.0`.
    :param accum_steps: Number of updates to accumulate before applying. This is used to simulate a higher batch size.
    :param lm_loss_coef: Language modeling loss coefficient -- a value between `0.0` - `1.0`
        that indicates how to trade off between language modeling loss
        and target model loss.  Usually not beneficial to turn on unless
        dataset size exceeds a few thousand examples.  Defaults to `0.0`.
    :param summarize_grads: Include gradient summary information in tensorboard.  Defaults to `False`.
    :param val_size: Validation set size as a percentage of all training data.  Validation will not be run by default if n_examples < 50.
        If n_examples > 50, defaults to max(5, min(100, 0.05 * n_examples))
    :param val_interval: Evaluate on validation set after `val_interval` batches.
        Defaults to 4 * val_size / batch_size to ensure that too much time is not spent on validation.
    :param lm_temp: Language model temperature -- a value of `0.0` corresponds to greedy maximum likelihood predictions
        while a value of `1.0` corresponds to random predictions. Defaults to `0.2`.
    :param seq_num_heads: Number of attention heads of final attention layer. Defaults to `16`.
    :param subtoken_predictions: Return predictions at subtoken granularity or token granularity?  Defaults to `False`.
    :param multi_label_sequences: Use a multi-labeling approach to sequence labeling to allow overlapping labels.
    :param multi_label_threshold: Threshold of sigmoid unit in multi label classifier.
        Can be increased or lowered to trade off precision / recall. Defaults to `0.5`.
    :param autosave_path: Save current best model (as measured by validation loss) to this location. Defaults to `None`.
    :param tensorboard_folder: Directory for tensorboard logs. Tensorboard logs will not be written
        unless tensorboard_folder is explicitly provided. Defaults to `None`.
    :param log_device_placement: Log which device each operation is placed on for debugging purposes.  Defaults to `False`.
    :param allow_soft_placement: Allow tf to allocate an operation to a different device if a device is unavailable.  Defaults to `True`.
    :param save_adam_vars: Save adam parameters when calling `model.save()`.  Defaults to `True`.
    :param num_layers_trained: How many layers to finetune.  Specifying a value less than 12 will train layers starting from model output. Defaults to `12`.
    :param train_embeddings: Should embedding layer be finetuned? Defaults to `True`.
    :param class_weights: One of 'log', 'linear', or 'sqrt'. Auto-scales gradient updates based on class frequency.  Can also be a dictionary that maps from true class name to loss coefficient. Defaults to `None`.
    :param oversample: Should rare classes be oversampled?  Defaults to `False`.
    :param params_device: Which device should gradient updates be aggregated on?
        If you are using a single GPU and have more than 4Gb of GPU memory you should set this to GPU PCI number (0, 1, 2, etc.). Defaults to `"cpu"`.
    :param eval_acc: if True, calculates accuracy and writes it to the tensorboard summary files for valudation runs.
    :param save_dtype: specifies what precision to save model weights with.  Defaults to `np.float32`.
    :param regression_loss: the loss to use for regression models. One of `L1` or `L2`, defaults to `L2`.
    :param prefit_init: if True, fit target model weigths before finetuning the entire model. Defaults to `False`.
    :param debugging_logs: if True, output tensorflow logs and turn off TQDM logging. Defaults to `False`.
    :param val_set: Where it is neccessary to use an explicit validation set, provide it here as a tuple (text, labels)
    :param per_process_gpu_memory_fraction: fraction of the overall amount of memory that each visible GPU should be allocated, defaults to `1.0`.
    """

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

        if attr == "base_model_path":
            full_path = finetune_model_path(self["base_model_path"])
            if os.path.exists(full_path):
                return full_path

        return self[attr]

    def __setitem__(self, key, value):
        if isinstance(value, GridSearchable):
            self.grid_searchable[key] = value.iterator
            value = value.default
        return super().__setitem__(key, value)

    def __setattr__(self, k, v):
        return self.__setitem__(k, v)

    __delattr__ = dict.__delitem__


def did_you_mean(keyword, keyword_pool):
    candidates = list(keyword_pool)
    closest_match_idx = np.argmin([
        edit_distance(keyword, candidate) for candidate in candidates
    ])
    return candidates[closest_match_idx]


def assert_valid_config(**kwargs):
    expected_keys = set(get_default_config().keys())
    for kwarg in kwargs:
        if kwarg not in expected_keys:
            raise FinetuneError(
                "Unexpected setting configuration: `{}` is an invalid keyword. "
                "Did you mean `{}`?".format(kwarg, did_you_mean(kwarg, expected_keys))
            )


def get_default_config():
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :return: Config object.
    """
    settings = Settings(
        # General Settings
        low_memory_mode=False,
        interpolate_pos_embed=False,
        save_adam_vars=True,
        shuffle_buffer_size=100,
        dataset_size=None,
        batch_size=2,
        visible_gpus=None, # defaults to all available
        n_epochs=GridSearchable(3, [1, 2, 3, 4]),
        seed=42,
        max_length=512,
        weight_stddev=0.02,
        save_dtype=None,
        val_set=None,
        per_process_gpu_memory_fraction=0.95,

        # Regularization
        embed_p_drop=0.1,
        attn_p_drop=0.1,
        resid_p_drop=0.1,
        clf_p_drop=0.1,
        l2_reg=GridSearchable(0.01, [0.0, 0.1, 0.01, 0.001]),
        vector_l2=False,

        # Early Stopping and Validation
        autosave_path=None,
        keep_best_model=False,
        early_stopping_steps=None,
        min_secs_between_eval=60,
        eval_acc=False,
        val_size=None,
        val_interval=None,

        # Debugging
        log_device_placement=False,
        soft_device_placement=True,
        tensorboard_folder=None,
        summarize_grads=False,
        debugging_logs=False,

        # Partial Fitting
        num_layers_trained=12,
        train_embeddings=True,

        # Class Imbalance
        class_weights=None,
        oversample=False,
        params_device="cpu",

        # Optimization Params
        optimizer="AdamW",
        b1=0.9,
        b2=0.999,
        epsilon=1e-8,
        lr_schedule='warmup_linear',
        lr=GridSearchable(6.25e-5, [6.25e-4, 6.25e-5, 6.25e-6]),
        lr_warmup=0.002,
        max_grad_norm=1.0,
        prefit_init=False,
        accum_steps=1,

        # MTL
        tasks=None,
        dont_optimize_zero_gradients=False,

        # Language Model Settings
        lm_loss_coef=0.0,
        lm_temp=0.2,

        # Sequence Labeling
        seq_num_heads=16,
        pad_token="<PAD>",
        pad_idx=None,
        subtoken_predictions=False,
        multi_label_sequences=False,
        multi_label_threshold=0.5,
        chunk_long_sequences=False,

        # Regression Params
        regression_loss="L2",

        # Association Params
        viable_edges=None,
        association_types=None,
        assocation_loss_weight=100.0,

        # Location of model weights
        base_model=GPTModel,
        base_model_path=None,

        # Possible `SourceModel` specific settings
        n_heads=None,
        n_layer=None,
        act_fn=None,
        n_embed=None,

        # for TextCNN SourceModel only
        kernel_sizes=None,
        num_filters_per_size=None,
        n_embed_featurizer=None, # needed because the dimensions CNN output are different from the embedding dimensions

        # BERT only
        bert_intermediate_size=None
    )
    return settings


def get_config(**kwargs):
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :param **kwargs: Keyword arguments to override default values.
    :return: Config object.    """
    assert_valid_config(**kwargs)
    config = get_default_config()
    config.base_model = kwargs.get('base_model', config.base_model)
    config.update(config.base_model.settings)
    config.update(kwargs)
    return config


def cpu_config(**kwargs):
    config = get_config(**kwargs)
    config.visible_gpus = []
    config.update(kwargs)
    return config

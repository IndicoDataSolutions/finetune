import logging
import os
import os.path
import warnings
from collections import namedtuple
from functools import lru_cache

import numpy as np
import tensorflow as tf
from nltk.metrics.distance import edit_distance

import finetune
from finetune.base_models import RoBERTa
from finetune.errors import FinetuneError

LOGGER = logging.getLogger("finetune")


def finetune_model_path(path):
    return os.path.abspath(
        os.path.join(os.path.dirname(finetune.__file__), "model", path)
    )


@lru_cache()
def all_gpus(visible_gpus=None):
    """
    Get integer ids of all available GPUs.
    """
    if visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible_gpus)
        return [int(gpu) for gpu in visible_gpus]
    devices = tf.config.experimental.list_physical_devices("GPU")
    LOGGER.info(
        " Visible GPUs: {{{}}}".format(
            ", ".join(
                [
                    "{}:{}".format(device.device_type, device.name).strip()
                    for device in devices
                ]
            )
        )
    )
    if not devices:
        warnings.warn("Failed to find available GPUS.  Falling back to CPU only mode.")
    device_ids = list(range(len(devices)))
    return device_ids


class Settings(dict):
    """
    Model configuration options

    :param base_model: Which base model to use - one of {GPT, GPT2, RoBERTa, BERT, TextCNN, TCN}, imported from finetune.base_models. Defaults to `RoBERTa`.
    :param batch_size: Number of examples per batch, defaults to `"auto"`.
    :param predict_batch_size: Batch size for prediction, defaults to `"auto"`.
    :param visible_gpus: List of integer GPU ids to spread out computation across, defaults to all available GPUs.
    :param n_epochs: Number of iterations through training data, defaults to `"auto"`.
    :param min_steps: Minimum number of steps a model will take when trained. Overrides n_epochs if epochs would result in a lower number of steps. Defaults to `None`.
    :param seed: Random seed to use for repeatability purposes, defaults to `42`.
    :param max_length: Maximum number of subtokens per sequence. Examples longer than this number will be truncated
        (unless `chunk_long_sequences=True` for SequenceLabeler models). Defaults to `"auto"`.
    :param weight_stddev: Standard deviation of initial weights. Defaults to `0.02`.
    :param save_dtype: Specifies what precision to save model weights with. Defaults to `None`.
    :param per_process_gpu_memory_fraction: Fraction of the overall amount of memory that each visible GPU should be allocated, defaults to `None`.
    :param xla: Uses TensorFlow XLA for compilation. Defaults to `False`.
    :param optimize_for: Optimize auto parameters for either `accuracy`, `speed`, or `predict_speed`. Defaults to `"accuracy"`.
    :param sort_by_length: Order the chunks by length to optimize padding usage. Defaults to `True`.
    :param collapse_whitespace: Any multiple adjacent spaces or tabs will be collapsed into one to improve token efficiency. Defaults to `False`.
    :param permit_uninitialized: Takes a regex to match to param paths that may be uninitialized. Usually set by the base model. Unlikely a user would need to override. Defaults to `None`.
    :param max_training_hours: Targets a maximum number of hours for the model to train rather than a number of steps. When set, learning rate as a function of time or steps whichever is running faster. Defaults to `None`.
    :param low_memory_mode: When True, only store partial gradients on forward pass
        and recompute remaining gradients incrementally in order to save memory. Defaults to `False`.
    :param float_16_predict: Whether to run prediction in float 16 mode, this is only available for bert based models and will likely only yield performance improvements on GPUs with native float16 support such as Volta and Tesla. Defaults to `"auto"`.
    :param mixed_precision: Whether to train in float16/32 mixed precision. Defaults to `"auto"`.
    :param shuffle_buffer_size: How many examples to load into a buffer before shuffling. Defaults to `100`.
    :param dataset_size: Must be specified in order to calculate the learning rate schedule when the inputs provided are generators rather than static datasets. Defaults to `None`.
    :param embed_p_drop: Embedding dropout probability. Defaults to `0.1`.
    :param attn_p_drop: Attention dropout probability. Defaults to `0.1`.
    :param resid_p_drop: Residual layer fully connected network dropout probability. Defaults to `0.1`.
    :param clf_p_drop: Classifier dropout probability. Defaults to `0.1`.
    :param l2_reg: L2 regularization coefficient. Defaults to `0.01`.
    :param vector_l2: Whether to apply weight decay regularization to vectors (biases, normalization etc..). Defaults to `False`.
    :param tensorboard_folder: Directory for tensorboard logs. Tensorboard logs will not be written
        unless tensorboard_folder is explicitly provided. Defaults to `None`.
    :param debugging_logs: If True, output tensorflow logs and turn off TQDM logging. Defaults to `False`.
    :param class_weights: One of 'log', 'linear', or 'sqrt'. Auto-scales gradient updates based on class frequency.
        Can also be a dictionary that maps from true class name to loss coefficient. Defaults to `None`.
    :param optimizer: Optimizer to use, current options include AdamW or AdamaxW. Defaults to `"AdamW"`.
    :param b1: Adam b1 parameter. Defaults to `0.9`.
    :param b2: Adam b2 parameter. Defaults to `0.999`.
    :param epsilon: Adam epsilon parameter. Defaults to `1e-8`.
    :param lr_schedule: Learning rate schedule -- see `finetune/optimizers.py` for more options. Defaults to `"warmup_linear"`.
    :param lr: Learning rate. Defaults to `"auto"`.
    :param lr_warmup: Learning rate warmup (percentage of all batches to warmup for). Defaults to `0.002`.
    :param max_grad_norm: Clip gradients larger than this norm. Defaults to `1.0`.
    :param accum_steps: Number of updates to accumulate before applying. This is used to simulate a higher batch size. Defaults to `1`.
    :param seq_num_heads: Number of attention heads of final attention layer. Defaults to `16`.
    :param pad_token: Set by the base models. Defaults to `"<PAD>"`.
    :param pad_idx: Set by the base models. Defaults to `None`.
    :param subtoken_predictions: Return predictions at subtoken granularity or token granularity? Defaults to `True`.
    :param chunk_long_sequences: When True, use a sliding window approach to predict on
        examples that are longer than max length. The progress bar will display the number of chunks processed rather than the number of examples. Defaults to `True`.
    :param chunk_context: How much context to include around chunked text. Defaults to `"auto"`.
    :param chunk_alignment: Alignment of the active section of the chunks "left", "right", "center". Defaults to `"center"`.
    :param add_eos_bos_to_chunk: Set by the base model and corresponds to whether we should have EOS and BOS tokens when we chunk long sequences. Defaults to `True`.
    :param filter_empty_examples: Only impacts training. Pretty self explanatory. Defaults to `False`.
    :param crf_sequence_labeling: Whether to use a CRF or not. If not we just use per-token softmax cross entropy. Defaults to `True`.
    :param max_empty_chunk_ratio: Controls the maximum ratio of empty to labeled chunks for sequence labeling. None includes all chunks, defaults to `1.0`.
    :param auto_negative_sampling: Method to use with long sparse documents to cut down on training
        time and limit false positives. Defaults to `False`.
    :param low_memory_ans: Trade off speed and memory when auto negative sampling is enabled. Defaults to `True`.
    :param max_document_chars: Maximum number of characters in a document before splitting into
        len(document) / max_document_chars "sub documents" for prediction to avoid memory issues
        during creation of the input pipeline. Defaults to `None` (no splitting).
    :param bio_tagging: Whether to use BIO tagging or not for sequence labeling. Defaults to `False`.
    :param base_model_path: Set by the base model. Defaults to `None`.
    :param n_heads: Base model specific parameter that controls the model construction. Defaults to `None`.
    :param n_layer: Base model specific parameter that controls the model construction. Defaults to `None`.
    :param act_fn: Base model specific parameter that controls the model construction. Defaults to `None`.
    :param n_embed: Base model specific parameter that controls the model construction. Defaults to `None`.
    :param n_filter: For TCN SourceModel only. Defaults to `None`.
    :param kernel_size: For TCN SourceModel only. Defaults to `None`.
    :param kernel_sizes: For TextCNN SourceModel only. Defaults to `None`.
    :param num_filters_per_size: For TextCNN SourceModel only. Defaults to `None`.
    :param n_embed_featurizer: Needed because the dimensions CNN output are different from the embedding dimensions. Defaults to `None`.
    :param bert_intermediate_size: BERT only. Defaults to `None`.
    :param bert_use_pooler: BERT only. Defaults to `True`.
    :param bert_use_type_embed: BERT only. Defaults to `True`.
    :param default_context: The default context to use for the model when no context is provided for a token. Defaults to `None`.
    :param context_dim: Number of context dimensions to be inserted. Defaults to `None`.
    :param context_injection: Whether to inject context into the model. Defaults to `False`.
    :param reading_order_removed: Set by the base model if the reading order is removed. Defaults to `False`.
    :param context_channels: Set by the base model, the number of channels used for doc rep X/Y position embeddings.
    :param norm_eps: ModernBERT parameter. Defaults to `1e-5`.
    :param mlp_p_drop: ModernBERT parameter. Defaults to `0.0`.
    :param global_attn_every_n_layers: ModernBERT parameter. Defaults to `3`.
    :param local_rope_theta: ModernBERT parameter. Defaults to `10000.0`.
    :param global_rope_theta: ModernBERT parameter. Defaults to `160000.0`.
    :param local_attention_window: ModernBERT parameter. Defaults to `128`.
    :param table_position: Whether to use absolute position encoding for cells in the TableModel. Defaults to `False`.
    :param table_position_type: Determines which types of position to use: "row_col" or "all". Defaults to `"row_col"`.
    :param include_row_col_summaries: Whether to include row-column summary representations that are distributed. Helpful in some cases when rows and columns are consistent. Defaults to `False`.
    :param down_project_feats: Whether to use a projection to bring the features back to the model size or not. Results in non-linear relationships between rows and columns when the target model is linear. Defaults to `False`.
    :param renorm_after_class_weights: Whether to reset the norm of the gradients or loss to be equivalent to what it was before class weights was applied. Renorming reduces the effectiveness of class weights especially with small batch sizes but improves stability and reduces likelihood of one bad sample blowing up the model. Defaults to `True`.
    :param max_row_col_embedding: Determines the maximum number of items in each row or column of the table. Defaults to `1024`.
    :param chunk_tables: Whether to chunk up tables within the featurizer when they exceed the max sizes of the featurizers. This is different from the Table Chunker within the TableLabeler util classes. Defaults to `False`.
    :param table_batching: Whether to use a table model specific batching approach that buckets by the size of the rows and columns. Results in more uniform GPU memory performance when using the table model. Defaults to `False`.
    :param reshuffle_chunks: At train time shuffles again after chunking so that for long documents the model does not see many batches from the same document. Defaults to `False`.
    :param predict_chunk_markers: Injects special prediction objects at sequence labeling prediction which outputs info on how the data was chunked up for the model. Defaults to `False`.
    :param version: Serialize finetune version with model. Defaults to `VERSION`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError

        if attr == "base_model_path":
            full_path = finetune_model_path(self["base_model_path"])
            if os.path.exists(full_path):
                return full_path

        return self[attr]

    def __setattr__(self, k, v):
        return self.__setitem__(k, v)

    __delattr__ = dict.__delitem__


def did_you_mean(keyword, keyword_pool):
    candidates = list(keyword_pool)
    closest_match_idx = np.argmin(
        [edit_distance(keyword, candidate) for candidate in candidates]
    )
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
    # lazy import to avoid circular dependency
    from finetune import VERSION

    settings = Settings(
        # General Settings
        low_memory_mode=False,
        float_16_predict="auto",
        mixed_precision="auto",
        shuffle_buffer_size=100,
        dataset_size=None,
        batch_size="auto",
        predict_batch_size="auto",
        visible_gpus=None,  # defaults to all available
        n_epochs="auto",
        min_steps=None,
        seed=42,
        max_length="auto",
        weight_stddev=0.02,
        save_dtype=None,
        per_process_gpu_memory_fraction=None,
        xla=False,
        optimize_for="accuracy",
        sort_by_length=True,
        collapse_whitespace=False,
        permit_uninitialized=None,
        max_training_hours=None,  # TODO: maybe we want to keep this?
        include_bos_eos=True,
        #
        # Regularization
        embed_p_drop=0.1,
        attn_p_drop=0.1,
        resid_p_drop=0.1,
        clf_p_drop=0.1,
        l2_reg=0.01,
        vector_l2=False,
        #        #
        # Debugging
        tensorboard_folder=None,
        debugging_logs=False,
        #
        # Class Imbalance
        class_weights=None,
        #
        # Optimization Params
        optimizer="AdamW",
        b1=0.9,
        b2=0.999,
        epsilon=1e-8,
        lr_schedule="warmup_linear",
        lr="auto",
        lr_warmup=0.002,
        max_grad_norm=1.0,
        accum_steps=1,
        #        #
        # Sequence Labeling
        seq_num_heads=16,
        pad_token="<PAD>",
        pad_idx=None,
        subtoken_predictions=True,
        chunk_long_sequences=True,
        chunk_context="auto",
        chunk_alignment="center",
        add_eos_bos_to_chunk=True,
        filter_empty_examples=False,
        crf_sequence_labeling=True,
        max_empty_chunk_ratio=1.0,
        auto_negative_sampling=False,
        low_memory_ans=True,
        max_document_chars=None,
        bio_tagging=False,
        #
        # Location of model weights
        base_model=RoBERTa,
        base_model_path=None,
        #
        # Possible `SourceModel` specific settings
        n_heads=None,
        n_layer=None,
        act_fn=None,
        n_embed=None,
        #
        # for TCN SourceModel only
        n_filter=None,
        kernel_size=None,
        #
        # for TextCNN SourceModel only
        kernel_sizes=None,
        num_filters_per_size=None,
        n_embed_featurizer=None,  # needed because the dimensions CNN output are different from the embedding dimensions
        #
        # BERT only
        bert_intermediate_size=None,
        bert_use_pooler=True,
        bert_use_type_embed=True,
        #
        # Auxiliary Information
        use_auxiliary_info=False,
        default_context=None,
        context_dim=None,  # number of context dimensions to be inserted
        #
        # Document Representation
        context_injection=False,
        reading_order_removed=False,
        anneal_reading_order=False,
        context_channels=None,
        #
        # ModernBERT
        norm_eps=1e-5,
        mlp_p_drop=0.0,
        global_attn_every_n_layers=3,
        local_rope_theta=10000.0,
        global_rope_theta=160000.0,
        local_attention_window=128,
        # Table model
        table_position=False,
        table_position_type="row_col",
        include_row_col_summaries=False,
        down_project_feats=False,
        renorm_after_class_weights=True,
        max_row_col_embedding=1024,
        chunk_tables=False,
        table_batching=False,
        # chunking_tweaks
        reshuffle_chunks=False,
        predict_chunk_markers=False,
        #
        # Serialize finetune version with model
        version=VERSION,
    )
    return settings


def get_config(error_on_invalid_keywords=True, **kwargs):
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :param **kwargs: Keyword arguments to override default values.
    :return: Config object."""
    if error_on_invalid_keywords:
        assert_valid_config(**kwargs)
    config = get_default_config()
    config.base_model = kwargs.get("base_model", config.base_model)
    config.update(config.base_model.settings)
    config.update(kwargs)
    return config


def cpu_config(**kwargs):
    config = get_config(**kwargs)
    config.visible_gpus = []
    config.update(kwargs)
    return config

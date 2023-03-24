import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import tensorflow as tf

from transformers import (
    ElectraTokenizerFast,
    ElectraConfig,
    TFBertMainLayer,
    BertTokenizerFast,
    BertConfig,
    XLMRobertaTokenizer,
    XLMRobertaConfig,
    T5Tokenizer,
    T5Config,
    AlbertTokenizerFast,
    AlbertConfig,
    BertTokenizer,
    LongformerTokenizerFast,
    LongformerConfig,
    DebertaV2Config,
    DebertaV2TokenizerFast,
)
from transformers.models.t5.modeling_tf_t5 import TFT5Model
from transformers.models.electra.modeling_tf_electra import TFElectraMainLayer
from transformers.models.roberta.modeling_tf_roberta import TFRobertaMainLayer
from transformers.models.albert.modeling_tf_albert import TFAlbertMainLayer
from transformers.models.longformer.modeling_tf_longformer import TFLongformerMainLayer
from transformers.models.deberta_v2.modeling_tf_deberta_v2 import TFDebertaV2MainLayer

from transformers.models.xlm_roberta.tokenization_xlm_roberta import (
    VOCAB_FILES_NAMES,
    PRETRAINED_VOCAB_FILES_MAP,
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
)

from finetune.util.huggingface_interface import finetune_model_from_huggingface

HFDebertaV3Base = finetune_model_from_huggingface(
    pretrained_weights="microsoft/deberta-v3-base",
    archive_map={"microsoft/deberta-v3-base": "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/tf_model.h5"},
    hf_featurizer=TFDebertaV2MainLayer,
    hf_tokenizer=DebertaV2TokenizerFast,
    hf_config=DebertaV2Config,
    config_overrides={"n_embed": 768, "n_epochs": 8, "lr": 1e-5, "batch_size": 2},
    weights_replacement=[
        ("tf_deberta_v2_model_4/deberta/embeddings", "model/featurizer/tf_deberta_v2_main_layer/embeddings"),
        ("tf_deberta_v2_model_4/deberta/encoder", 'model/featurizer/tf_deberta_v2_main_layer/encoder'),
        ("tf_deberta_v2_base_model/deberta", "model/featurizer/encoder"),

    ]
)

HFXLMRoberta = finetune_model_from_huggingface(
    pretrained_weights="jplu/tf-xlm-roberta-base",
    archive_map={
        "jplu/tf-xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-xlm-roberta-base/tf_model.h5"
    },
    hf_featurizer=TFRobertaMainLayer,
    hf_tokenizer=XLMRobertaTokenizer,
    hf_config=XLMRobertaConfig,
    weights_replacement=[
        (
            "tfxlm_roberta_for_masked_lm/roberta",
            "model/featurizer/tf_roberta_main_layer",
        )
    ],
)

# FIXME: - Currently very broken. 
# Variable scoping has changed in a very weird way internally to huggingface.
# Note that this new variable scoping breaks out of our top level model/featurizer
# scope so the Checks inside saver do not capture it.
# Additionally, there are issues with != vs tf.math.not_equal that are possible to
# patch on the forwards pass, but something strange happens on the backwards pass
# And we hit what I believe is an equivalent error. Big sad :(

# HFLongformer = finetune_model_from_huggingface(
#     pretrained_weights="allenai/longformer-base-4096",
#     archive_map={
#         "allenai/longformer-base-4096": "https://cdn.huggingface.co/allenai/longformer-base-4096/tf_model.h5"
#     },
#     hf_featurizer=TFLongformerMainLayer,
#     hf_tokenizer=LongformerTokenizerFast,
#     hf_config=LongformerConfig,
#     weights_replacement=[
#         (
#             "tf_longformer_for_masked_lm/longformer",
#             "model/featurizer/tf_longformer_main_layer",
#         )
#     ],
#     config_overrides={
#         "low_memory_mode": True,
#         "batch_size": 2,
#         "n_epochs": 16,
#         "lr": 1.68e-3,
#         "max_grad_norm": 1.1,
#         "lr_warmup": 0.7,  # Not a typo - just weird.
#         "l2_reg": 0.194,
#     },
#     required_patches=["longformer"]
# )


HFElectraGen = finetune_model_from_huggingface(
    pretrained_weights="google/electra-base-generator",
    archive_map={
        "google/electra-base-generator": "https://cdn.huggingface.co/google/electra-base-generator/tf_model.h5"
    },
    hf_featurizer=TFElectraMainLayer,
    hf_tokenizer=ElectraTokenizerFast,
    hf_config=ElectraConfig,
    weights_replacement=[
        ("tf_electra_for_masked_lm/electra", "model/featurizer/tf_electra_main_layer")
    ],
)

HFElectraDiscrim = finetune_model_from_huggingface(
    pretrained_weights="google/electra-base-discriminator",
    archive_map={
        "google/electra-base-discriminator": "https://cdn.huggingface.co/google/electra-base-discriminator/tf_model.h5"
    },
    hf_featurizer=TFElectraMainLayer,
    hf_tokenizer=ElectraTokenizerFast,
    hf_config=ElectraConfig,
    weights_replacement=[
        (
            "tf_electra_for_pre_training/electra",
            "model/featurizer/tf_electra_main_layer",
        )
    ],
)

HFBert = finetune_model_from_huggingface(
    pretrained_weights="bert-base-uncased",
    archive_map={
        "bert-base-uncased": "https://cdn.huggingface.co/bert-base-uncased-tf_model.h5"
    },
    hf_featurizer=TFBertMainLayer,
    hf_tokenizer=BertTokenizerFast,
    hf_config=BertConfig,
    weights_replacement=[
        ("tf_bert_for_pre_training_2/bert/", "model/featurizer/tf_bert_main_layer/"),
        ("tf_bert_for_pre_training/bert/", "model/featurizer/tf_bert_main_layer/"),
    ],
)

HFT5 = finetune_model_from_huggingface(
    pretrained_weights="t5-base",
    archive_map={"t5-base": "https://cdn.huggingface.co/t5-base-tf_model.h5"},
    hf_featurizer=TFT5Model,
    hf_tokenizer=T5Tokenizer,
    hf_config=T5Config,
    weights_replacement=[
        ("tf_t5with_lm_head_model/shared/", "model/featurizer/shared/shared/"),
        ("tf_t5with_lm_head_model/encoder", "model/featurizer/encoder"),
        ("tf_t5with_lm_head_model/decoder", "model/target/decoder"),
    ],
    include_bos_eos="eos",
    add_tokens=["{", "}", "<"],  # "[", "]"],
    required_patches=["t5"]
)

HFT5Small = finetune_model_from_huggingface(
    pretrained_weights="t5-small",
    archive_map={"t5-small": "https://cdn.huggingface.co/t5-small-tf_model.h5"},
    hf_featurizer=TFT5Model,
    hf_tokenizer=T5Tokenizer,
    hf_config=T5Config,
    weights_replacement=[
        ("tf_t5with_lm_head_model/shared/", "model/featurizer/shared/shared/"),
        ("tf_t5with_lm_head_model/encoder", "model/featurizer/encoder"),
        ("tf_t5with_lm_head_model/decoder", "model/target/decoder"),
    ],
    include_bos_eos="eos",
    add_tokens=["{", "}", "<"],  # "[", "]"],
    required_patches=["t5"]
)

HFAlbert = finetune_model_from_huggingface(
    pretrained_weights="albert-base-v2",
    archive_map={
        "albert-base-v2": "https://cdn.huggingface.co/albert-base-v2-tf_model.h5"
    },
    hf_featurizer=TFAlbertMainLayer,
    hf_tokenizer=AlbertTokenizerFast,
    hf_config=AlbertConfig,
    weights_replacement=[
        ("tf_albert_for_masked_lm_1/albert/", "model/featurizer/tf_albert_main_layer/")
    ],
    config_overrides={"n_embed": 768, "n_epochs": 8, "lr": 2e-5, "batch_size": 8},
    aggressive_token_alignment=True,
)


HFAlbertXLarge = finetune_model_from_huggingface(
    pretrained_weights="albert-xlarge-v2",
    archive_map={
        "albert-xlarge-v2": "https://cdn.huggingface.co/albert-xlarge-v2-tf_model.h5"
    },
    hf_featurizer=TFAlbertMainLayer,
    hf_tokenizer=AlbertTokenizerFast,
    hf_config=AlbertConfig,
    weights_replacement=[
        ("tf_albert_for_masked_lm_5/albert/", "model/featurizer/tf_albert_main_layer/")
    ],
    config_overrides={"n_embed": 2048, "n_epochs": 8, "lr": 1e-5, "batch_size": 2},
    aggressive_token_alignment=True,
)

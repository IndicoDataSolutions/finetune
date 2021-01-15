import logging
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)

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
)
from transformers.modeling_tf_t5 import TFT5Model
from transformers.modeling_tf_electra import TFElectraMainLayer
from transformers.modeling_tf_roberta import TFRobertaMainLayer
from transformers.modeling_tf_albert import TFAlbertMainLayer
from transformers.tokenization_xlm_roberta import (
    VOCAB_FILES_NAMES,
    PRETRAINED_VOCAB_FILES_MAP,
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
)

from finetune.util.huggingface_interface import finetune_model_from_huggingface


HFXLMRoberta = finetune_model_from_huggingface(
    pretrained_weights="jplu/tf-xlm-roberta-base",
    archive_map={
        "jplu/tf-xlm-roberta-base": "https://cdn.huggingface.co/bert/jplu/tf-xlm-roberta-base/tf_model.h5"
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
    include_bos_eos=False,
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
)

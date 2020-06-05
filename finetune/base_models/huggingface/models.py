from transformers import (
    TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP,
    ElectraTokenizerFast,
    ElectraConfig,

    TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    TFBertMainLayer,
    BertTokenizerFast,
    BertConfig
)
from transformers.modeling_tf_electra import TFElectraMainLayer

from finetune.util.huggingface_interface import finetune_model_from_huggingface


HFElectraGen = finetune_model_from_huggingface(
    pretrained_weights="google/electra-base-generator",
    archive_map=TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP,
    hf_featurizer=TFElectraMainLayer,
    hf_tokenizer=ElectraTokenizerFast,
    hf_config=ElectraConfig,
    weights_replacement=[
        ("tf_electra_for_masked_lm/electra", "model/featurizer/tf_electra_main_layer")
    ],
)

HFElectraDiscrim = finetune_model_from_huggingface(
    pretrained_weights="google/electra-base-discriminator",
    archive_map=TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP,
    hf_featurizer=TFElectraMainLayer,
    hf_tokenizer=ElectraTokenizerFast,
    hf_config=ElectraConfig,
    weights_replacement=[
        ("tf_electra_for_pre_training/electra", "model/featurizer/tf_electra_main_layer")
    ],
)

HFBert = finetune_model_from_huggingface(
    pretrained_weights="bert-base-uncased",
    archive_map=TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    hf_featurizer=TFBertMainLayer,
    hf_tokenizer=BertTokenizerFast,
    hf_config=BertConfig,
    weights_replacement=[
        ("tf_bert_for_pre_training_2/bert/", "model/featurizer/tf_bert_main_layer/"),
        ("tf_bert_for_pre_training/bert/", "model/featurizer/tf_bert_main_layer/"),
    ],
)

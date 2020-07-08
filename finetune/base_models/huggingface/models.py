from transformers import (
    TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP,
    ElectraTokenizerFast,
    ElectraConfig,

    TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    TFBertMainLayer,
    BertTokenizerFast,
    BertConfig,

    XLMRobertaTokenizer,
    XLMRobertaConfig,
)
from transformers.modeling_tf_electra import TFElectraMainLayer
from transformers.modeling_tf_roberta import TFRobertaMainLayer
from transformers.tokenization_xlm_roberta import (
    VOCAB_FILES_NAMES, PRETRAINED_VOCAB_FILES_MAP, PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
)
from transformers.tokenization_utils import PreTrainedTokenizerFast
from tokenizers import SentencePieceBPETokenizer

from finetune.util.huggingface_interface import finetune_model_from_huggingface


class XLMRobertaTokenizerFast(XLMRobertaTokenizer, PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_vocab_files_map['merges_file'] = {"xlm-roberta-base": "/path"}
    pretrained_init_configuration = {"xlm-roberta": {"merges_file": "/path1"}}
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        PreTrainedTokenizerFast.__init__(
            self,
            SentencePieceBPETokenizer(
                vocab_file=vocab_file,
                unk_token=unk_token,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use XLMRobertaTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1

        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}



HFXLMRoberta = finetune_model_from_huggingface(
    pretrained_weights="jplu/tf-xlm-roberta-base",
    archive_map={'jplu/tf-xlm-roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-xlm-roberta-large/tf_model.h5"},
    hf_featurizer=TFRobertaMainLayer,
    # hf_tokenizer=XLMRobertaTokenizerFast,
    hf_tokenizer=XLMRobertaTokenizer,
    hf_config=XLMRobertaConfig,
    weights_replacement=[
        ("tfxlm_roberta_for_masked_lm/roberta", "model/featurizer/tf_roberta_main_layer")
    ],
)


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

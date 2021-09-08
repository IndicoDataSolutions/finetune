import os

import numpy as np

import finetune
from finetune.encoding.input_encoder import EncodedOutput, BaseEncoder
from finetune.base_models.bert.tokenizer import FullTokenizer

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, "model", "bert", "vocab.txt")
VOCAB_PATH_MULTILINGUAL = os.path.join(
    FINETUNE_FOLDER, "model", "bert", "vocab_multi.txt"
)
VOCAB_PATH_LARGE = os.path.join(FINETUNE_FOLDER, "model", "bert", "vocab_large.txt")
VOCAB_PATH_DISTILBERT = os.path.join(
    FINETUNE_FOLDER, "model", "bert", "distillbert_vocab.txt"
)
VOCAB_PATH_LAYOUTLM = os.path.join(
    FINETUNE_FOLDER, "model", "bert", "layoutlm_vocab.txt"
)


class BERTEncoder(BaseEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """

    UNK_IDX = 0

    def __init__(self, encoder_path=None, vocab_path=VOCAB_PATH, lower_case=False):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)
        self.lower_case = lower_case

        self.tokenizer = FullTokenizer(
            vocab_file=self.vocab_path, do_lower_case=self.lower_case
        )

        self.start_token = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.delimiter_token = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.mask_token = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        self.end_token = self.delimiter_token

    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab)

    def _token_length(self, token):
        return len(token.strip().replace("##", ""))

    def _encode(self, texts):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        batch_tokens = []
        batch_token_idxs = []
        batch_char_ends = []
        batch_char_starts = []
        for i, text in enumerate(texts):
            subtokens, token_starts, token_ends = self.tokenizer.tokenize(text)
            subtoken_idxs = self.tokenizer.convert_tokens_to_ids(subtokens)

            batch_tokens.append(subtokens)
            batch_token_idxs.append(subtoken_idxs)
            batch_char_ends.append(token_ends)
            batch_char_starts.append(token_starts)

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            token_ends=batch_char_ends,
            token_starts=batch_char_starts,
        )

    def decode(self, ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """

        raise NotImplementedError(
            "The inverse of the BERT encoder is not implemented because we have not implemented generation with BERT"
        )


class BERTEncoderLarge(BERTEncoder):
    def __init__(
        self, encoder_path=None, vocab_path=VOCAB_PATH_LARGE, lower_case=False
    ):
        super().__init__(
            encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case
        )


class BERTEncoderMultuilingal(BERTEncoder):
    def __init__(
        self, encoder_path=None, vocab_path=VOCAB_PATH_MULTILINGUAL, lower_case=False
    ):
        super().__init__(
            encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case
        )


class DistilBERTEncoder(BERTEncoder):
    def __init__(
        self, encoder_path=None, vocab_path=VOCAB_PATH_DISTILBERT, lower_case=True
    ):
        super().__init__(
            encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case
        )


class LayoutLMEncoder(BERTEncoder):
    def __init__(
        self, encoder_path=None, vocab_path=VOCAB_PATH_LAYOUTLM, lower_case=True
    ):
        super().__init__(
            encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case
        )

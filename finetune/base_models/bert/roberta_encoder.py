import json
import os

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

import finetune
from finetune.base_models.gpt2 import encoder as gpt2_encoder
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
DICT_PATH = os.path.join(FINETUNE_FOLDER, "model", "bert", "dict.txt")
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, "model", "bert", "roberta_encoder.json")
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, "model", "bert", "roberta_vocab.bpe")


class RoBERTaEncoder(GPT2Encoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """

    offset = 4

    def __init__(
        self, encoder_path=gpt2_encoder.ENCODER_PATH, vocab_path=gpt2_encoder.VOCAB_PATH
    ):
        self.freqs = {}
        index = 0
        with open(DICT_PATH, "r", encoding="utf-8") as freq_dict:
            lines = freq_dict.readlines()
            for line in lines:
                idx = line.rfind(" ")
                if idx == -1:
                    raise ValueError(
                        "Incorrect dictionary format, expected '<token> <cnt>'"
                    )
                if "madeupword" in line[:idx]:
                    index += 1
                    continue
                token_idx = int(line[:idx])
                self.freqs[token_idx + self.offset] = (
                    index + self.offset
                )  # add 4 for the special tokens at beginning
                index += 1
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

    def _convert_to_embed_idx(self, idx):
        return self.freqs[idx]

    def _add_extra_toks(self):
        self.special_tokens = []
        self.encoder["<BOS>"] = 0
        self.encoder["<PAD>"] = 1
        self.encoder["<EOS>"] = 2
        self.encoder["<UNK>"] = 3
        self.freqs[3] = 3
        self.start_token = 0  # bos from roberta
        self.delimiter_token = 2  # eos from roberta
        self.end_token = 2  # eos from roberta
        self.UNK_IDX = 3  # unk from roberta
        # If base model file doesn't contain a mask token, use <UNK> token idx
        self.mask_token = self.encoder.get("<mask>", 3)


class RoBERTaEncoderSlow(RoBERTaEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    Now with support for MLM objective
    """

    offset = 4

    def _convert_to_embed_idx(self, idx):
        return idx

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=VOCAB_PATH):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)


class RoBERTaEncoderV2(BaseEncoder):
    offset = 4

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=VOCAB_PATH):

        with open(vocab_path, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        with open(encoder_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=bpe_merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.RobertaProcessing(
            sep=("</s>", vocab["</s>"]),
            cls=("<s>", vocab["<s>"]),
            add_prefix_space=False,
            trim_offsets=True,  # True by default on Roberta (historical)
        )

        self.start_token = 0  # bos from roberta
        self.delimiter_token = 2  # eos from roberta
        self.end_token = 2  # eos from roberta
        self.UNK_IDX = 3  # unk from roberta
        self.mask_token = 50264
        self.mapping = {
            vocab["<s>"]: 0,
            vocab["</s>"]: 2,
            vocab["<unk>"]: 3,
        }
        self.initialized = True

    @property
    def vocab_size(self):
        return 50269

    def _encode(self, texts):
        batch_tokens = []
        batch_token_idxs = []
        batch_char_ends = []
        batch_char_starts = []
        for i, text in enumerate(texts):
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            batch_token_idxs.append(
                [
                    i + self.offset if i not in self.mapping else self.mapping[i]
                    for i in encoded.ids
                ]
            )
            token_ends = []
            token_starts = []
            tokens = []
            for (start, end), t in zip(encoded.offsets, encoded.tokens):
                if t.startswith("Ġ") or t.startswith("Ċ") or t.startswith("Â"):
                    t = t[1:]
                tokens.append(t.strip())
                if token_ends:
                    # tokenizers outputs start and end tokens that duplicate in the case of
                    #  a single char mapping to multiple tokens.
                    start = max(token_ends[-1], start)
                if end - start > len(t):
                    start = end - len(t)
                token_starts.append(start)
                token_ends.append(end)

            batch_tokens.append(tokens)
            batch_char_ends.append(token_ends)
            batch_char_starts.append(token_starts)
        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            token_ends=batch_char_ends,
            token_starts=batch_char_starts,
        )

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class RoBERTaEncoderXDoc(RoBERTaEncoderV2):
    offset = 0

    @property
    def vocab_size(self):
        return 50265

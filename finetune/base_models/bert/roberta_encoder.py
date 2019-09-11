import os
import json
import regex as re

import numpy as np

import finetune
from finetune.encoding.input_encoder import EncodedOutput
from finetune.base_models.gpt2.encoder import GPT2Encoder, bytes_to_unicode, ENCODER_PATH, VOCAB_PATH

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
DICT_PATH = os.path.join(FINETUNE_FOLDER, "model", "bert", "dict.txt")


class RoBERTaEncoder(GPT2Encoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=VOCAB_PATH):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)
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
                    break
                token_idx = int(line[:idx])
                self.freqs[str(token_idx + 4)] = (
                    index + 4
                )  # add 4 for the special tokens at beginning
                index += 1

    def _lazy_init(self, errors="replace"):
        if self.initialized:
            return

        # Load encoder
        with open(self.encoder_path, "r") as f:
            self.encoder = json.load(f)
            # shift all indices forward four places to make place for embeddings for start, pad, delim, clf
            self.encoder.update((token, idx + 4) for token, idx in self.encoder.items())

        # Load BPE
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [
            tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]
        ]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.special_tokens = []
        self.encoder["<BOS>"] = 0
        self.encoder["<PAD>"] = 1
        self.encoder["<EOS>"] = 2
        self.encoder["<UNK>"] = 3
        self.start_token = 0  # bos from roberta
        self.delimiter_token = 2  # eos from roberta
        self.end_token = 2  # eos from roberta
        self.UNK_IDX = 3  # unk from roberta

        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.initialized = True

    def _encode(self, texts, labels=None):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        self._lazy_init()
        batch_tokens = []
        batch_token_idxs = []
        batch_label_idxs = []
        batch_char_ends = (
            []
        )  # to account for the fact that some BPEs have different lengths than their original tokens (e.g. special characters such as bullets)
        batch_char_starts = []
        label = None
        offset = (
            0
        )  # tracks offset between this fields' character_locs, which start at 0, and the 'start' keys in context which track the entire document (not just this field)

        skipped = 0

        for i, text in enumerate(texts):
            if labels is not None:
                label = labels[i]

            subtokens = []
            subtoken_idxs = []
            char_ends = []
            char_starts = []
            token_start = 0

            tokens = re.findall(self.pat, text)
            if not tokens:
                offset += len(text)  # for spans that are just whitespace
                skipped += 1
                continue
            i -= skipped
            for j, token in enumerate(tokens):
                # token = token.strip()
                encoded_token = "".join(
                    self.byte_encoder[b] for b in token.encode("utf-8")
                )
                bpe_toks = self.bpe(encoded_token).split(" ")
                try:
                    if token.strip():
                        token_start = text.index(token, token_start)
                except ValueError:
                    # text_standardization oddity
                    traceback.print_exc()
                    continue

                subtokens.extend(bpe_toks)
                subtoken_idxs.extend(
                    [self.encoder.get(t, self.UNK_IDX) for t in bpe_toks]
                )

                token_char_starts = [token_start] * len(bpe_toks)

                if np.sum([len(tok) for tok in bpe_toks]) > len(token):
                    token_char_ends = (
                        np.asarray([len(token.strip()) for tok in bpe_toks])
                        + token_start
                    )
                else:
                    token_char_ends = (
                        np.cumsum([len(tok) for tok in bpe_toks]) + token_start
                    )

                token_start += len(token.strip())
                char_ends.extend(token_char_ends)
                char_starts.extend(token_char_starts)

            batch_tokens.append(subtokens)
            for k in range(len(subtoken_idxs)):
                subtoken_idxs[k] = self.freqs[str(subtoken_idxs[k])]
            batch_token_idxs.append(subtoken_idxs)
            batch_char_ends.append(char_ends)
            batch_char_starts.append(char_starts)
            if labels is not None:
                batch_label_idxs.append([label] * len(subtoken_idxs))

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            char_locs=batch_char_ends,
            char_starts=batch_char_starts,
        )
import os
import json
import traceback
import regex as re
from functools import lru_cache

import numpy as np

import finetune
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput, get_pairs

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, "model", "gpt2", "encoder.json")
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, "model", "gpt2", "vocab.bpe")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class GPT2Encoder(BaseEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """

    UNK_IDX = 0
    offset = 0

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=VOCAB_PATH):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

        # Load encoder
        with open(self.encoder_path, "r") as f:
            self.encoder = json.load(f)

        if self.offset != 0:
            self.encoder.update(
                (token, idx + self.offset) for token, idx in self.encoder.items()
            )

        # Load BPE
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [
            tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]
        ]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self._add_extra_toks()

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = "replace"
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def _add_extra_toks(self):
        self.special_tokens = ["_delimiter_", "_classify_"]
        for token in self.special_tokens:
            self.encoder[token] = len(self.encoder)
        self.start_token = self.encoder["<|endoftext|>"]
        self.delimiter_token = self.encoder["_delimiter_"]
        self.end_token = self.encoder["_classify_"]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _convert_to_embed_idx(self, idx):
        return idx

    def _decode_token(self, bpe_toks):
        decoded_bpe_toks = []
        for i, bpe in enumerate(bpe_toks):
            temp_toks = bpe
            st = None
            offset = 1
            while st is None:
                try:
                    st = bytes(self.byte_decoder[c] for c in temp_toks).decode("utf-8")
                except UnicodeDecodeError:
                    if len(bpe_toks[i + offset]) == 0:
                        offset += 1
                    temp_toks += bpe_toks[i + offset][0]
                    bpe_toks[i + offset] = bpe_toks[i + offset][1:]
            decoded_bpe_toks.append(st)
        return decoded_bpe_toks

    def _encode(self, texts):
        """
        Convert a sample of raw text to a list of byte-pair encoded token indices.
        """
        batch_tokens = []
        batch_token_idxs = []
        batch_char_ends = []
        # to account for the fact that some BPEs have different lengths than their original tokens
        # (e.g. special characters such as bullets)
        batch_char_starts = []

        for i, text in enumerate(texts):  # text = one label span
            subtokens = []
            subtoken_idxs = []
            char_ends = []
            char_starts = []
            token_start = 0

            tokens = re.findall(self.pat, text)
            for j, token in enumerate(tokens):
                encoded_token = "".join(
                    self.byte_encoder[b] for b in token.encode("utf-8")
                )
                bpe_toks = self.bpe(encoded_token).split(" ")
                decoded_bpe_toks = self._decode_token(bpe_toks)
                try:
                    if token.strip():
                        token_start = text.index(token.strip(), token_start)
                except ValueError:
                    # text_standardization oddity
                    traceback.print_exc()
                    continue

                subtokens.extend(decoded_bpe_toks)
                subtoken_idxs.extend(
                    [self.encoder.get(t, self.UNK_IDX) for t in bpe_toks]
                )
                lens = [None for _ in bpe_toks]

                for i, tok in enumerate(decoded_bpe_toks):
                    lens[i] = len(tok.strip())

                token_char_ends = np.cumsum(lens) + token_start
                token_char_starts = [token_start] + token_char_ends[:-1].tolist()
                token_start += len(token.strip())
                char_ends.extend(token_char_ends)
                char_starts.extend(token_char_starts)

            batch_tokens.append(subtokens)

            for k in range(len(subtoken_idxs)):
                subtoken_idxs[k] = self._convert_to_embed_idx(subtoken_idxs[k])

            batch_token_idxs.append(subtoken_idxs)
            batch_char_ends.append(char_ends)
            batch_char_starts.append(char_starts)
        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            token_ends=batch_char_ends,
            token_starts=batch_char_starts,
        )

    def decode(self, token_ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """
        text = "".join([self.decoder[token_id] for token_id in token_ids])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text

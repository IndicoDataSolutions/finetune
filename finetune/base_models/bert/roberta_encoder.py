import os
import json
import regex as re
from functools import lru_cache

import numpy as np

import finetune
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput, get_pairs

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, "model", "gpt2", "encoder.json")
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, "model", "gpt2", "vocab.bpe")
DICT_PATH = os.path.join(FINETUNE_FOLDER, "model", "bert", "dict.txt")


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


class RoBERTaEncoder(BaseEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """

    UNK_IDX = 0

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
        self.start = 0  # bos from roberta
        self.delimiter = 2  # eos from roberta
        self.clf_token = 2  # eos from roberta
        self.UNK_IDX = 3  # unk from roberta

        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.initialized = True

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

    def _encode(self, texts, labels=None, context=None):
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
        batch_context = []
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

            # Context is tokenwise, so we need to duplicate contexts for each subtoken of a token, and to match length of labels
            if context is not None:
                text_context = self.line_up_context(
                    context, batch_char_ends[i], batch_tokens[i], subtoken_idxs, offset
                )
                batch_context.extend(text_context)
                offset += batch_char_ends[i][-1]

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            char_locs=batch_char_ends,
            char_starts=batch_char_starts,
            context=batch_context
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

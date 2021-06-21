"""
Convert plain text to format accepted by model (token idxs + special tokens).
"""
import re
import json
import os
import warnings
import codecs

import numpy as np
from ftfy.fixes import uncurl_quotes

import finetune
from finetune.encoding.input_encoder import (
    get_spacy,
    EncodedOutput,
    BaseEncoder,
    get_pairs,
)


FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, "model", "gpt", "encoder.json")
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, "model", "gpt", "vocab.bpe")
SUBS = {"—": "-", "–": "-", "―": "-", "…": "...", "´": "'"}


def _text_standardize(text):
    """
    Fixes some issues the spacy tokenizer had on books corpus
    Also handles whitespace standardization
    """
    text = re.sub(
        """(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""",
        r" \1 ",
        text,
    )
    text = re.sub("\s*\n\s*", " \n ", text)
    text = re.sub("[^\S\n]+", " ", text)
    return uncurl_quotes(text.strip().lower())


class GPTEncoder(BaseEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """

    UNK_IDX = 0

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=VOCAB_PATH):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

        with open(self.encoder_path, "r") as f:
            self.encoder = json.load(f)

        with codecs.open(self.vocab_path, encoding="utf8"):
            merges = (
                codecs.open(self.vocab_path, encoding="utf8").read().split("\n")[1:-1]
            )
            merges = [tuple(merge.split()) for merge in merges]
            self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.special_tokens = ["_start_", "_delimiter_", "_classify_"]
        for token in self.special_tokens:
            self.encoder[token] = len(self.encoder)

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.start_token = self.encoder["_start_"]
        self.delimiter_token = self.encoder["_delimiter_"]
        self.end_token = self.encoder["_classify_"]

    def _token_length(self, token):
        return len(token.strip().replace("</w>", ""))

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

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
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def _encode(self, texts):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        nlp = get_spacy()

        batch_tokens = []
        batch_token_idxs = []
        batch_char_ends = []
        # to account for the fact that some BPEs have different lengths than their original tokens
        # (e.g. special characters such as bullets)

        batch_char_starts = []
        for i, text in enumerate(texts):

            raw_text = text.lower()

            # Only fine to apply this fix because it preserves character locations
            ftfy_text = uncurl_quotes(raw_text)
            tokens = nlp(_text_standardize(text))
            subtokens = []
            subtoken_idxs = []
            char_starts = []
            char_ends = []
            token_start = 0

            for j, token in enumerate(tokens):
                bpe_toks = self.bpe(token.text).split(" ")

                try:
                    stripped_text = token.text.strip()
                    if stripped_text:
                        token_start = ftfy_text.index(stripped_text, token_start)
                except ValueError:
                    warnings.warn(
                        "Failed to find token `{}` in text.".format(token.text)
                    )
                    continue

                subtokens.extend(bpe_toks)
                subtoken_idxs.extend(
                    [self.encoder.get(SUBS.get(t, t), self.UNK_IDX) for t in bpe_toks]
                )

                assert len("".join(bpe_toks).replace("</w>", "")) == len(
                    token.text.replace(" ", "")
                )

                token_char_ends = (
                    np.cumsum(
                        [len(tok.strip().replace("</w>", "")) for tok in bpe_toks]
                    )
                    + token_start
                )
                token_char_starts = [token_start] + token_char_ends[:-1].tolist()
                token_start += len(token.text.strip())
                char_ends.extend(token_char_ends)
                char_starts.extend(token_char_starts)

            batch_tokens.append(subtokens)
            batch_token_idxs.append(subtoken_idxs)
            batch_char_ends.append(char_ends)
            batch_char_starts.append(char_starts)

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

        return "".join(
            [self.decoder.get(word_idx, "<unk>") for word_idx in ids]
        ).replace("</w>", " ")


def aggregate_to_full_tokens(attn, tokens, token_starts, token_ends, attention=True):
    to_combine = []
    spacy_attn = []
    spacy_token_starts = []
    spacy_token_ends = []
    spacy_start = None
    for token, prob, start, end in zip(tokens, attn, token_starts, token_ends):
        to_combine.append(prob)
        if spacy_start is None:
            spacy_start = start
        if token.endswith("</w>"):
            if attention:
                spacy_attn.append(np.max(to_combine, 0))
            else:
                spacy_attn.append(to_combine[-1])
            spacy_token_starts.append(spacy_start)
            spacy_token_ends.append(end)
            to_combine = []
            spacy_start = None

    if attention:
        spacy_attn = spacy_attn / sum(spacy_attn)
        key = "attention_weights"
    else:
        key = "explanation"

    return {
        key: spacy_attn,
        "token_starts": spacy_token_starts,
        "token_ends": spacy_token_ends,
    }


def finetune_to_indico_explain(raw_texts, attn_weights, encoder, attention=True):
    """
    Maps the attention weights or explain probability values one-to-one with the raw text tokens.

    :param raw_texts: A list of segmented text of the form list(list(str))
    :param attn_weights: An array of attention weights of shape [batch, seq_len]
    :param encoder: The encoder used in the model that output the attention weights
    :param attention: A boolean flag, true for inputting attention matrices, false for explain probabilities.
    :return: A list of dictionaries, each with the following keys:
    - 'attention_weights': A list of attention weights for each token in the text
    - 'token_starts': A list of the start tokens for each spacy token in the text
    - 'token_ends': A list of the end tokens for each spacy token in the text

    """
    spacy_outputs = []

    encoded_docs = encoder._encode(raw_texts)

    for doc_idx, (raw_text) in enumerate(raw_texts):
        tokens = encoded_docs.tokens[doc_idx]
        token_ends = encoded_docs.token_ends[doc_idx]
        token_starts = encoded_docs.token_ends[doc_idx]
        if attention:
            # offset of one to take into account the start token
            clf_token_idx = len(tokens) + 1
            # take the average values over the attention heads for the attention weights of the classify token
            attn = np.mean(attn_weights[doc_idx], axis=0)[clf_token_idx][
                1:clf_token_idx
            ]  # [num_tokens]
        else:
            # There is no start token
            clf_token_idx = len(tokens)
            attn = attn_weights[doc_idx][:clf_token_idx]
        # map one-to-one with spacy tokenization
        spacy_output = aggregate_to_full_tokens(
            attn, tokens, token_starts, token_ends, attention=attention
        )
        spacy_outputs.append(spacy_output)
    return spacy_outputs

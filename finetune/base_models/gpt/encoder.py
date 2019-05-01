"""
Convert plain text to format accepted by model (token idxs + special tokens).
"""
import re
import json
import os
import warnings
import functools
from collections import namedtuple
import codecs

import ftfy
import spacy
import numpy as np
import tensorflow as tf

import finetune
from finetune.encoding.input_encoder import NLP, EncodedOutput, BaseEncoder, get_pairs


FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, 'model', 'gpt', 'encoder.json')
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, 'model', 'gpt', 'vocab.bpe')
SUBS = {
    '—': '-',
    '–': '-',
    '―': '-',
    '…': '...',
    '´': "'"
}


def _text_standardize(text):
    """
    Fixes some issues the spacy tokenizer had on books corpus
    Also handles whitespace standardization
    """
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return ftfy.fix_text(text.strip().lower())


class GPTEncoder(BaseEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """
    UNK_IDX = 0

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=VOCAB_PATH):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

    def _lazy_init(self):
        # Must set
        if self.initialized:
            return

        with open(self.encoder_path, 'r') as f:
            self.encoder = json.load(f)

        with codecs.open(self.vocab_path, encoding='utf8'):
            merges = codecs.open(self.vocab_path, encoding='utf8').read().split('\n')[1:-1]
            merges = [tuple(merge.split()) for merge in merges]
            self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.special_tokens = ['_start_', '_delimiter_', '_classify_']
        for token in self.special_tokens:
            self.encoder[token] = len(self.encoder)

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.cache = {}
        self.start = self.encoder['_start_']
        self.delimiter = self.encoder['_delimiter_']
        self.clf_token = self.encoder['_classify_']
        self.initialized = True

    def _token_length(self, token):
        return len(token.strip().replace('</w>', ''))

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
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
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def _encode(self, texts, labels=None):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        self._lazy_init()
        batch_tokens = []
        batch_token_idxs = []
        batch_label_idxs = []
        batch_character_locs = []
        label = None

        for i, text in enumerate(texts):
            if labels is not None:
                label = labels[i]
            raw_text = text.lower()
            tokens = NLP(_text_standardize(text))
            subtokens = []
            subtoken_idxs = []
            tok_pos = []
            token_start = 0

            for j, token in enumerate(tokens):
                bpe_toks = self.bpe(token.text).split(' ')

                try:
                    if token.text.strip():
                        token_start = raw_text.index(token.text.strip(), token_start)
                except ValueError:
                    # text_standardization oddity
                    continue

                subtokens.extend(bpe_toks)
                subtoken_idxs.extend([
                    self.encoder.get(SUBS.get(t, t), self.UNK_IDX)
                    for t in bpe_toks
                ])

                assert len("".join(bpe_toks).replace("</w>", "")) == len(token.text.replace(' ', ''))
                subtoken_positions = np.cumsum([len(tok.replace("</w>", '')) for tok in bpe_toks]) + token_start

                token_start += len(token.text.strip())

                tok_pos.extend(subtoken_positions)

            batch_tokens.append(subtokens)
            batch_token_idxs.append(subtoken_idxs)
            batch_character_locs.append(tok_pos)
            if labels is not None:
                batch_label_idxs.append([label] * len(subtoken_idxs))

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            char_locs=batch_character_locs,
        )

    def decode(self, ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """

        return "".join([self.decoder.get(word_idx, '<unk>') for word_idx in ids]).replace("</w>", " ")


def to_spacy_attn(attn, tokens, token_starts, token_ends):
    to_combine = []
    spacy_attn = []
    spacy_token_starts = []
    spacy_token_ends = []
    spacy_start = None
    for token, prob, start, end in zip(tokens, attn, token_starts, token_ends):
        to_combine.append(prob)
        if not spacy_start:
            spacy_start = start
        if token.endswith('</w>'):
            spacy_attn.append(max(to_combine))
            spacy_token_starts.append(spacy_start)
            spacy_token_ends.append(end)
            to_combine = []
            spacy_start = None
    spacy_attn = spacy_attn/sum(spacy_attn)
    return {
        'attention_weights': spacy_attn,
        'token_starts': spacy_token_starts,
        'token_ends': spacy_token_ends
    }


def finetune_to_indico_attention_weights(raw_texts, attn_weights, encoder):
    """
    Maps the attention weights one-to-one with the raw text tokens.
    
    :param raw_texts: A list of segmented text of the form list(list(str))
    :param attn_weights: An array of attention weights of shape [batch, seq_len]
    :param encoder: The encoder used in the model that output the attention weights
    :return: A list of dictionaries, each with the following keys:
    - 'attention_weights': A list of attention weights for each token in the text
    - 'token_starts': A list of the start tokens for each spacy token in the text
    - 'token_ends': A list of the end tokens for each spacy token in the text

    """
    spacy_outputs = []

    encoded_docs = encoder._encode(raw_texts)

    for doc_idx, (raw_text) in enumerate(raw_texts):
        tokens = encoded_docs.tokens[doc_idx]
        token_ends = encoded_docs.char_locs[doc_idx]
        token_lengths = [encoder._token_length(token) for token in tokens]
        token_starts = [end - length for end, length in zip(token_ends, token_lengths)]
        # offset of one to take into account the start token
        clf_token_idx = len(tokens) + 1
        # take the average values over the attention heads for the attention weights of the classify token 
        attn = np.mean(attn_weights[doc_idx], axis=0)[clf_token_idx][1:clf_token_idx]  # [num_tokens]
        # map one-to-one with spacy tokenization
        spacy_output = to_spacy_attn(attn, tokens, token_starts, token_ends)
        spacy_outputs.append(spacy_output)
    return spacy_outputs

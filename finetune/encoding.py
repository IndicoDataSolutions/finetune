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

from finetune.config import PAD_TOKEN

ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model/encoder_bpe_40000.json')
BPE_PATH = os.path.join(os.path.dirname(__file__), 'model/vocab_40000.bpe')
NLP = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])

EncodedOutput = namedtuple("EncodedOutput", [
    "token_ids", # list of list of subtoken ids (ints)
    "tokens",    # list of list of subtokens (strs)
    "labels",    # list of list of labels 
    "char_locs", # list of list of character locations (ints)
])
EncodedOutput.__new__.__defaults__ = (None,) * len(EncodedOutput._fields)
ArrayEncodedOutput = namedtuple("ArrayEncodedOutput", [
    "token_ids", # int array shape (batch, seq_length)
    "tokens",    # list of list of subtokens (str) passed through from `EncoderOutput`
    "labels",    # object array shape (batch, seq_length)
    "char_locs", # list of list of char_locs (int) passed through from `EncoderOutput`
    "mask",      # int array shape (batch, seq_length)
])
ArrayEncodedOutput.__new__.__defaults__ = (None,) * len(ArrayEncodedOutput._fields)

SUBS = {
    '—': '-',
    '–': '-',
    '―': '-',
    '…': '...',
    '´': "'"
}


def _flatten(nested_lists):
    return functools.reduce(lambda x, y: x + y, nested_lists, [])


def _get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def _text_standardize(text):
    """
    Fixes some issues the spacy tokenizer had on books corpus
    Also handles whitespace standardization
    """
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return ftfy.fix_text(text.strip().lower())


class TextEncoder(object):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """
    UNK_IDX = 0

    def __init__(self):
        self.encoder = json.load(open(ENCODER_PATH))
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.special_tokens = ['_start_', '_delimiter_', '_classify_']
        for token in self.special_tokens:
            self.encoder[token] = len(self.encoder)

        self.decoder = {v: k for k, v in self.encoder.items()}

        merges = codecs.open(BPE_PATH, encoding='utf8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.start = self.encoder['_start_']
        self.delimiter = self.encoder['_delimiter_']
        self.clf_token = self.encoder['_classify_']

    @property
    def vocab_size(self):
        return len(self.encoder)

    def __getitem__(self, key):
        return self.encoder[key]

    def __setitem__(self, key, value):
        self.encoder[key] = value

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = _get_pairs(word)

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
                pairs = _get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def _encode(self, texts, labels=None, verbose=True):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
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
                        token_start = raw_text.index(token.text, token_start)
                except:
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

    def _cut_and_concat(self, *, encoded, max_length, verbose, special_tokens=None, start=None, delimiter=None,
                        end=None):
        """
        Takes some tokenized text and arranges it into a format that maximises the amount of kept text from each
        whilst keeping the overall sequence length within max_length tokens. It also adds the 3 special tokens. Start,
         Classify and Delimiter.
        :param encoded: Lists of shape [batch, n_fields, num_tokens]
        :param max_length: Int representing the max length of a single sample
        :param verbose: Bool of whether to print the TQDM bar or not.
        :param start: Override the default start token.
        :param delimiter: Override the default delimiter token.
        :param end: Override the default classify token
        :return: Formatted outputs of the form. [batch, num_tokens] where num_tokens' <= max_length
        """
        start = start or special_tokens or self.start
        delimiter = delimiter or special_tokens or self.delimiter
        clf_token = end or special_tokens or self.clf_token

        num_samples = len(encoded)
        adjusted_max_length = max_length - num_samples - 1
        allocated_max_len = adjusted_max_length // num_samples

        overflows = [allocated_max_len - len(sequence) for sequence in encoded]
        spare = sum(overflows)

        if spare >= 0:
            cut_len = None
        else:
            warnings.warn("Document is longer than max length allowed, trimming document to {} tokens.".format(
                max_length
            ))
            empty_tokens = sum(max(overflow, 0) for overflow in overflows)
            num_over = [max(overflow, 0) for overflow in overflows].count(0)
            if num_over == 0:
                cut_len = allocated_max_len
            else:
                cut_len = allocated_max_len + (empty_tokens // num_over)

        joined = [start]
        for d in encoded:
            joined += (d[:cut_len] + [delimiter])
        joined = joined[:-1] + [clf_token]

        return joined

    def encode_multi_input(self, Xs, Y=None, max_length=None, verbose=True, pad_token=PAD_TOKEN):
        """
        Encodes the text for passing to the model, also tracks the location of each token to allow reconstruction.
        It can also, optionally, construct a per-token labels as required for training.
        :param Xs: A list of lists of string -- [n_fields, n_segments]
        :param Y: A list of list of targets -- [n_batch, n_segments]
        :param max_length: Max length of the sequences.
        :param verbose: Flag to set whether to output a status bar.
        :return: A Labeled Sequence Object.
        """

        token_ids = []
        tokens = []
        positions = []
        labels = []

        # for each field in that example
        for field in Xs:
            assert isinstance(field, (list, tuple)), "This should be a list of strings, if its not," \
                "you've done something wrong... instead it's {}".format(tf.contrib.framework.nest.map_structure(type, field))
            encoded = self._encode(field, labels=Y)
            token_ids.append(_flatten(encoded.token_ids))
            tokens.append(_flatten(encoded.tokens))
            positions.append(_flatten(encoded.char_locs))
            labels.append(_flatten(encoded.labels))
            if len(tokens[-1]) > (max_length - 2):
                warnings.warn(
                    "Some examples are longer than the max_length. Please trim documents or increase `max_length`. "
                    "Fallback behaviour is to use the first {} byte-pair encoded tokens".format(max_length - 2)
                )

        # merge fields + truncate if necessary
        token_ids = self._cut_and_concat(
            encoded=token_ids,
            max_length=max_length,
            verbose=verbose
        )
        tokens = self._cut_and_concat(
            encoded=tokens,
            max_length=max_length,
            verbose=verbose
        )
        locations = self._cut_and_concat(
            encoded=positions,
            max_length=max_length,
            verbose=verbose,
            special_tokens=-1
        )

        if Y is None:
            labels = None
        else:
            labels = self._cut_and_concat(
                encoded=labels,
                max_length=max_length,
                verbose=verbose,
                special_tokens=pad_token
            )

        return EncodedOutput(
            token_ids=token_ids,
            tokens=tokens,
            labels=labels,
            char_locs=locations,
        )

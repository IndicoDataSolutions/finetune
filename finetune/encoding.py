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


def get_pairs(word):
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


def _flatten(nested_lists):
    return functools.reduce(lambda x, y: x + y, nested_lists, [])


class BaseEncoder(object):
    """
    Base class for GPT and GPT-2 encoding
    Translates raw texts into structured token arrays
    """
    UNK_IDX = 0

    def __init__(self, encoder_path, vocab_path):
        self.initialized = False
        self.encoder_path = encoder_path
        self.vocab_path = vocab_path

        # Required public attributes -- consider refactor to prevent requiring direct
        # access of these attributes
        self.special_tokens = None
        self.start = None
        self.delimiter = None
        self.clf_token = None
        self.encoder = None
        self.decoder = None

    @property
    def vocab_size(self):
        self._lazy_init()
        return len(self.encoder)

    def __getitem__(self, key):
        return self.encoder[key]

    def __setitem__(self, key, value):
        self.encoder[key] = value

    def _encode(self, texts, labels=None):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        raise NotImplementedError

    def decode(self, ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """

        raise NotImplementedError

    def _cut_and_concat(self, *, encoded, max_length, special_tokens=None, start=None, delimiter=None,
                        end=None):
        """
        Takes some tokenized text and arranges it into a format that maximises the amount of kept text from each
        whilst keeping the overall sequence length within max_length tokens. It also adds the 3 special tokens. Start,
         Classify and Delimiter.
        :param encoded: Lists of shape [batch, n_fields, num_tokens]
        :param max_length: Int representing the max length of a single sample
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

    def _token_length(self, token):
        return len(token)

    def encode_multi_input(self, Xs, Y=None, max_length=None, pad_token=None):
        """
        Encodes the text for passing to the model, also tracks the location of each token to allow reconstruction.
        It can also, optionally, construct a per-token labels as required for training.
        :param Xs: A list of lists of string -- [n_fields, n_segments]
        :param Y: A list of list of targets -- [n_batch, n_segments]
        :param max_length: Max length of the sequences.
        :return: A Labeled Sequence Object.
        """

        token_ids = []
        tokens = []
        positions = []
        labels = []

        # for each field in that example
        for field in Xs:
            assert isinstance(field, (list, tuple)), "This should be a list of strings, instead it's {}".format(
                tf.contrib.framework.nest.map_structure(type, field)
            )
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
            max_length=max_length
        )
        tokens = self._cut_and_concat(
            encoded=tokens,
            max_length=max_length
        )
        locations = self._cut_and_concat(
            encoded=positions,
            max_length=max_length,
            special_tokens=-1
        )

        if Y is None:
            labels = None
        else:
            labels = self._cut_and_concat(
                encoded=labels,
                max_length=max_length,
                special_tokens=pad_token
            )

        return EncodedOutput(
            token_ids=token_ids,
            tokens=tokens,
            labels=labels,
            char_locs=locations,
        )

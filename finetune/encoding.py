"""
Convert plain text to format accepted by model (token idxs + special tokens)
"""
import re
import json
import os
import warnings
import functools
from collections import namedtuple

import ftfy
import spacy
import numpy as np
from tqdm import tqdm

from finetune.config import PAD_TOKEN

ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model/encoder_bpe_40000.json')
BPE_PATH = os.path.join(os.path.dirname(__file__), 'model/vocab_40000.bpe')

EncoderOutput = namedtuple("EncoderOutput", ["token_ids", "labels", "char_locs"])
LabeledSequence = namedtuple("LabeledSequence", ["token_ids", "labels", "char_locs"])


def flatten(nested_lists):
    return functools.reduce(lambda x, y: x + y, nested_lists, [])


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


def text_standardize(text):
    """
    Fixes some issues the spacy tokenizer had on books corpus
    Also handles whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')

    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return ftfy.fix_text(text.strip().lower())


class TextEncoder(object):
    """
    Mostly a wrapper for a public python BPE tokenizer
    """
    UNK_IDX = 0

    def __init__(self):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(ENCODER_PATH))
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.special_tokens = ['_start_', '_delimiter_', '_classify_']
        for token in self.special_tokens:
            self.encoder[token] = len(self.encoder)

        self.decoder = {v: k for k, v in self.encoder.items()}

        merges = open(BPE_PATH).read().split('\n')[1:-1]
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

    def _encode(self, texts, labels=None, verbose=True):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        batch_token_idxs = []
        batch_label_idxs = []
        batch_character_locs = []
        label = None
        for i, text in tqdm(enumerate(texts), ncols=80, leave=False, disable=(not verbose)):
            if labels:
                label = labels[i]
            raw_text = text.lower()
            tokens = self.nlp(text_standardize(text))
            subtoken_idxs = []
            tok_pos = []
            token_start = 0
            for token in tokens:
                bpe_toks = self.bpe(token.text).split()
                subtoken_idxs.extend([
                    self.encoder.get(t, self.UNK_IDX)
                    for t in bpe_toks
                ])
                token_start = raw_text.find(token.text, token_start)
                assert len("".join(bpe_toks).replace("</w>", "")) == len(token.text.strip())
                subtoken_positions = np.cumsum([len(tok.replace("</w>", '')) for tok in bpe_toks]) + token_start
                token_start += len(token.text)
                tok_pos.extend(subtoken_positions)
            batch_token_idxs.append(subtoken_idxs)
            batch_character_locs.append(tok_pos)
            if labels is not None:
                batch_label_idxs.append([label] * len(subtoken_idxs))

        return EncoderOutput(batch_token_idxs, batch_label_idxs, batch_character_locs)

    def decode(self, ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """

        return "".join([self.decoder.get(word_idx, '<unk>') for word_idx in ids]).replace("</w>", " ")

    def trim(self, seqs, max_length, start=None, end=None):
        # account for start + end tokens
        if not start:
            start = self.start
        if not end:
            end = self.clf_token
        adjusted_max_length = max_length - 2

        if any([len(seq) > adjusted_max_length for seq in seqs]):
            warnings.warn("Document is longer than max length allowed, trimming document to {} tokens.".format(
                max_length
            ))
        seqs = [
            [start] + seq[:adjusted_max_length] + [end]
            for seq in seqs
        ]
        return seqs

    def encode_for_classification(self, texts, max_length, verbose=True):
        """
        Convert a batch of raw text to btye-pair encoded token indices,
        and add appropriate special tokens to match expected model input
        """
        batch_token_idxs = self._encode(texts, verbose=verbose).token_ids
        return self.trim(batch_token_idxs, max_length=max_length)

    def encode_for_comparison(self, texts, max_length, verbose=True):
        pass

    def encode_multi_input(self, *Xs, max_length, verbose=True):
        encoded = [self._encode(x).token_ids for x in Xs]
        return self._cut_and_concat(encoded=encoded, max_length=max_length, verbose=verbose)

    def _cut_and_concat(self, *, encoded, max_length, verbose, special_tokens=None, start=None, delimiter=None,
                        end=None):
        """
        Takes some tokenized text and arranges it into a format that maximises the amount of kept text from each
        whilst keeping the overall sequence length within max_length tokens. It also adds the 3 special tokens. Start,
         Classify and Delimiter.
        :param encoded: Lists of shape [sequences, batch, num_tokens]
        :param max_length: Int representing the max length of a single sample
        :param verbose: Bool of whether to print he TQDM bar or not.
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
        outputs = []
        for single_datum in tqdm(zip(*encoded), disable=not verbose):
            overflows = [allocated_max_len - len(sequence) for sequence in single_datum]
            spare = sum(overflows)
            if spare >= 0:
                cut_len = None
            else:
                empty_tokens = sum(max(overflow, 0) for overflow in overflows)
                num_over = [min(overflow, 0) for overflow in overflows].count(0)
                if num_over == 0:
                    cut_len = allocated_max_len
                else:
                    cut_len = allocated_max_len + (empty_tokens // num_over)
            joined = [start]
            for d in single_datum:
                joined += (d[:cut_len] + [delimiter])
            joined = joined[:-1] + [clf_token]
            outputs.append(joined)
        return outputs

    def encode_sequence_labeling(self, X, Y=None, max_length=None, verbose=True):
        # X: [n_batch, seq_len]
        # Y: [n_batch, seq_len]
        tokens = []
        labels = []
        positions = []

        for i, x in enumerate(X):
            targets = None if Y is None else Y[i]
            encoded = self._encode(x, labels=targets)
            positions.append(flatten(encoded.char_locs))
            tokens.append(flatten(encoded.token_ids))
            labels.append(flatten(encoded.labels))

        tokens = self._cut_and_concat(
            encoded=[tokens],
            max_length=max_length,
            verbose=verbose
        )

        locations = self._cut_and_concat(
            encoded=[positions],
            max_length=max_length,
            verbose=verbose,
            special_tokens=-1
        )

        if Y is None:
            labels = None
        else:
            labels = self._cut_and_concat(
                encoded=[labels],
                max_length=max_length,
                verbose=verbose,
                special_tokens=PAD_TOKEN
            )

        return LabeledSequence(
            token_ids=tokens,
            char_locs=locations,
            labels=labels
        )

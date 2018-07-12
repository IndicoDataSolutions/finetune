"""
Convert plain text to format accepted by model (token idxs + special tokens)
"""
import re
import json
import os
import warnings
import functools

import numpy as np

import ftfy
import spacy
from tqdm import tqdm

ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model/encoder_bpe_40000.json')
BPE_PATH = os.path.join(os.path.dirname(__file__), 'model/vocab_40000.bpe')

PAD_LABEL = "<PAD>"


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

    def _encode(self, texts, verbose=True):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        batch_token_idxs = []
        batch_label_idxs = []
        batch_character_locs = []
        label = None
        if verbose:
            texts = tqdm(texts, ncols=80, leave=False)

        for text in texts:
            if type(text) != str and len(text) == 2:
                text, label = text

            raw_text = text.lower()
            text = self.nlp(text_standardize(text))
            token_idxs = []
            tok_pos = []
            for token in text:
                bpe_toks = self.bpe(token.text).split()
                token_idxs.extend([
                    self.encoder.get(t, self.UNK_IDX)
                    for t in bpe_toks
                ])
                token_idx = raw_text.find(token.text, tok_pos[-1] if tok_pos else 0)
                assert len("".join(bpe_toks).replace("</w>", "")) == len(token), "{}, {}".format(bpe_toks, token)
                token_positions = np.cumsum([len(tok.replace("</w>", '')) for tok in bpe_toks]) + token_idx

                tok_pos.extend(token_positions)
            batch_token_idxs.append(token_idxs)
            batch_character_locs.append(tok_pos)
            if label is not None:
                batch_label_idxs.append([label] * len(token_idxs))

        return batch_token_idxs, batch_label_idxs, batch_character_locs

    def encode_for_classification(self, texts, max_length, verbose=True):
        """
        Convert a batch of raw text to btye-pair encoded token indices,
        and add appropriate special tokens to match expected model input
        """
        batch_token_idxs, *_ = self._encode(texts, verbose=verbose)
        # account for start + end tokens
        adjusted_max_length = max_length - 2
        if any([len(token_idxs) > adjusted_max_length for token_idxs in batch_token_idxs]):
            warnings.warn("Document is longer than max length allowed, trimming document to {} tokens.".format(
                max_length
            ))
        batch_token_idxs = [
            [self.start] + token_idxs[:adjusted_max_length] + [self.clf_token]
            for token_idxs in batch_token_idxs
        ]
        return batch_token_idxs

    def encode_for_entailment(self, question, answer, max_length, verbose=True):
        question_ids, *_ = self._encode(question)
        answer_ids, *_ = self._encode(answer)
        return self._cut_and_concat([question_ids, answer_ids], max_length, verbose)

    def encode_multi_input(self, *Xs, max_length, verbose=True):
        encoded = [self._encode(x)[0] for x in Xs]
        return self._cut_and_concat(encoded, max_length, verbose)

    def _cut_and_concat(self, encoded, max_length, verbose, start=None, delimiter=None, classify=None):
        """
        Takes some tokenized text and arranges it into a format that maximises the amount of kept text from each
        whilst keeping the overall sequence length within max_length tokens. It also adds the 3 special tokens. Start,
         Classify and Delimiter.
        :param encoded: Lists of shape [sequences, batch, num_tokens]
        :param max_length: Int representing the max length of a single sample
        :param verbose: Bool of whether to print he TQDM bar or not.
        :param start: Override the default start token.
        :param delimiter: Override the default delimiter token.
        :param classify: Override the default classify token
        :return: Formatted outputs of the form. [batch, num_tokens'] where num_tokens' <= max_length
        """
        start = start or self.start
        delimiter = delimiter or self.delimiter
        clf_token = classify or self.clf_token
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

    def encode_input_sequence_labeling(self, *Xs, max_length, verbose=True):
        # Xs = [n_inputs, n_batch, n_items_in_seq, 2]
        text = []
        labels_out = []
        for input in Xs:
            input_text = []
            input_labels = []
            for batch in input:
                tokens, labels, _ = self._encode(batch)
                tokens_single = functools.reduce(lambda x, y: x + y, tokens, [])
                labels_single = functools.reduce(lambda x, y: x + y, labels, [])
                input_text.append(tokens_single)
                input_labels.append(labels_single)
            text.append(input_text)
            labels_out.append(input_labels)

        return (self._cut_and_concat(text, max_length, verbose),
                self._cut_and_concat(labels_out, max_length, verbose, PAD_LABEL, PAD_LABEL, PAD_LABEL))

    def encode_sequence_labeling_inferrence(self, *Xs, max_length, verbose=True):
        # Xs = [n_inputs, n_batch, n_items_in_seq, 2]
        text = []
        positions_out = []
        for input in Xs:
            print(input)
            tokens, _, token_positions = self._encode(input)
            text.append(tokens)
            positions_out.append(token_positions)

        return (self._cut_and_concat(text, max_length, verbose),
                self._cut_and_concat(positions_out, max_length, verbose, -1, -1, -1))

"""
Convert plain text to format accepted by model (token idxs + special tokens)
"""
import re
import json
import os
import warnings

import ftfy
import spacy
from tqdm import tqdm

ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model/encoder_bpe_40000.json')
BPE_PATH = os.path.join(os.path.dirname(__file__), 'model/vocab_40000.bpe')


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

    def encode(self, texts, verbose=True):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        batch_token_idxs = []

        for text in tqdm(texts, ncols=80, leave=False, disable=(not verbose)):
            text = self.nlp(text_standardize(text))
            token_idxs = []
            for token in text:
                token_idxs.extend([
                    self.encoder.get(t, self.UNK_IDX)
                    for t in self.bpe(token.text).split()
                ])
            batch_token_idxs.append(token_idxs)
        return batch_token_idxs

    def encode_for_classification(self, texts, max_length, verbose=True):
        """
        Convert a batch of raw text to btye-pair encoded token indices,
        and add appropriate special tokens to match expected model input
        """
        batch_token_idxs = self.encode(texts, verbose=verbose)
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

    def encode_for_comparison(self, texts, max_length, verbose=True):
        pass

    def encode_for_entailment(self, question, answer, max_length, verbose=True):
        question_ids = self.encode(question, verbose=verbose)
        answer_ids = self.encode(answer, verbose=verbose)
        adjusted_max_length = max_length - 3

        half_max_len = adjusted_max_length // 2  # Initial allocation for question
        question_answer_pairs = []
        for qid, aid in zip(question_ids, answer_ids):
            q = len(qid)
            a = len(aid)
            spare = max(0, half_max_len - min(q, a))
            q_adj = min(q, half_max_len + spare)
            a_adj = min(a, half_max_len + spare)

            question_answer_pairs.append([self.start] + qid[:q_adj] + [self.delimiter] + aid[:a_adj] + [self.clf_token])

        return question_answer_pairs

    def encode_multi_input(self, *Xs, max_length, verbose=True):
        encoded = [self.encode(x, verbose=verbose) for x in Xs]
        num_samples = len(encoded)
        adjusted_max_length = max_length - num_samples - 1
        allocated_max_len = adjusted_max_length // num_samples
        outputs = []
        for single_datum in zip(*encoded):
            overflows = [allocated_max_len - len(sequence) for sequence in single_datum]
            spare = sum(overflows)
            if spare >= 0:
                cut_len = None
            else:
                empty_tokens = sum(max(overflow, 0) for overflow in overflows)
                num_over = [min(overflow, 0) for overflow in overflows].count(0)
                cut_len = allocated_max_len + (empty_tokens // num_over)
            joined = [self.start]
            for d in single_datum:
                joined += (d[:cut_len] + [self.delimiter])
            joined = joined[:-1] + [self.clf_token]
            outputs.append(joined)
        return outputs

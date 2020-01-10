"""
Convert plain text to format accepted by model (token idxs + special tokens).
"""
import warnings
import functools
from collections import namedtuple, Counter

import spacy
import tensorflow as tf
import numpy as np


NLP = spacy.load("en", disable=["parser", "tagger", "ner", "textcat"])

EncodedOutput = namedtuple(
    "EncodedOutput",
    [
        "token_ids",  # list of list of subtoken ids (ints)
        "tokens",  # list of list of subtokens (strs)
        "labels",  # list of list of labels
        "char_locs",  # list of list of character locations (ints)
        "char_starts",  # list of list of character starts (locs are character ends) (ints)
    ],
)
EncodedOutput.__new__.__defaults__ = (None,) * len(EncodedOutput._fields)
ArrayEncodedOutput = namedtuple(
    "ArrayEncodedOutput",
    [
        "token_ids",  # int array shape (batch, seq_length)
        "tokens",  # list of list of subtokens (str) passed through from `EncoderOutput`
        "labels",  # object array shape (batch, seq_length)
        "char_locs",  # list of list of char_locs (int) passed through from `EncoderOutput`
        "mask",  # int array shape (batch, seq_length)
    ],
)
ArrayEncodedOutput.__new__.__defaults__ = (None,) * len(ArrayEncodedOutput._fields)

SUBS = {"—": "-", "–": "-", "―": "-", "…": "...", "´": "'"}


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
    Base class for text encoders.
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
        self.start_token = None
        self.delimiter_token = None
        self.end_token = None
        self.encoder = None
        self.decoder = None

    def _lazy_init(self):
        pass

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

    def _cut_and_concat(
        self,
        *,
        encoded,
        max_length,
        special_tokens=None,
        start=None,
        delimiter=None,
        end=None
    ):
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
        start = start or special_tokens or self.start_token
        delimiter = delimiter or special_tokens or self.delimiter_token
        clf_token = end or special_tokens or self.end_token

        num_samples = len(encoded)
        adjusted_max_length = max_length - num_samples - 1
        allocated_max_len = adjusted_max_length // num_samples

        overflows = [allocated_max_len - len(sequence) for sequence in encoded]
        spare = sum(overflows)

        if spare >= 0:
            cut_len = None
        else:
            warnings.warn(
                "Document is longer than max length allowed, trimming document to {} tokens.".format(
                    max_length
                )
            )
            empty_tokens = sum(max(overflow, 0) for overflow in overflows)
            num_over = [max(overflow, 0) for overflow in overflows].count(0)
            if num_over == 0:
                cut_len = allocated_max_len
            else:
                cut_len = allocated_max_len + (empty_tokens // num_over)

        joined = [start]
        for d in encoded:
            joined += d[:cut_len] + [delimiter]
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
            assert isinstance(
                field, (list, tuple)
            ), "This should be a list of strings, instead it's {} {}".format(
                tf.contrib.framework.nest.map_structure(type, field), Xs
            )
            encoded = self._encode(field, labels=Y)
            token_ids.append(_flatten(encoded.token_ids))
            tokens.append(_flatten(encoded.tokens))
            positions.append(_flatten(encoded.char_locs))
            labels.append(_flatten(encoded.labels))
            if len(tokens[-1]) > (max_length - 2):
                warnings.warn(
                    "Some examples are longer than the max_length. Please trim documents or increase `max_length`. "
                    "Fallback behaviour is to use the first {} byte-pair encoded tokens".format(
                        max_length - 2
                    )
                )

        # merge fields + truncate if necessary
        token_ids = self._cut_and_concat(encoded=token_ids, max_length=max_length)
        tokens = self._cut_and_concat(encoded=tokens, max_length=max_length)
        locations = self._cut_and_concat(
            encoded=positions, max_length=max_length, special_tokens=-1
        )

        if Y is None:
            labels = None
        else:
            labels = self._cut_and_concat(
                encoded=labels, max_length=max_length, special_tokens=pad_token
            )

        return EncodedOutput(
            token_ids=token_ids,
            tokens=tokens,
            labels=labels,
            char_locs=locations,
        )

    def __setstate__(self, state):
        self.__init__()

    def __getstate__(self):
        return {"Encoder": None}


def tokenize_context(context, encoded_output, config):
    """ Tokenize the context corresponding to a single sequence of text """
    seq_len = len(encoded_output.token_ids)
    context_keys = list(k for k in sorted(context[0].keys()) if k not in ['token', 'start', 'end'])
    context_by_char_loc = sorted([(c['end'], [c[k] for k in context_keys]) for c in context], key=lambda c: c[0])
    # default context is the sequence majority
    default_context = [config.default_context[k] for k in context_keys]
    current_char_loc = 0
    tokenized_context = []
    for char_loc in encoded_output.char_locs:
        # Note: this assumes that the tokenization will never lump multiple tokens into one
        if char_loc == -1:
            tokenized_context.append(default_context)
        else:
            if char_loc > context_by_char_loc[current_char_loc][0]:
                current_char_loc += 1
                if current_char_loc >= len(context_by_char_loc): 
                    raise ValueError("Context cannot be fully matched as it appears to not cover the end of the sequence")

            tokenized_context.append(context_by_char_loc[current_char_loc][1])
    # padded value doesn't matter since it will be masked out
    expanded_context = np.pad(tokenized_context, ((0, seq_len - len(tokenized_context)), (0, 0)), 'constant')
    assert len(expanded_context) == len(encoded_output.token_ids)
    return expanded_context

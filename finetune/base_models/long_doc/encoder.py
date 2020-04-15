import os
import json
import regex as re
import logging
from functools import lru_cache

import numpy as np

import finetune
import spacy
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput, get_pairs

import unicodedata

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
LOGGER = logging.getLogger("finetune")

class LongDocEncoder(BaseEncoder):
    UNK_IDX = 0

    def __init__(self, encoder_path=None, vocab_path=None):
        self.nlp = None
        pass

    def _lazy_init(self):
        self.nlp = spacy.load("en")
        self.start_token = np.zeros([300], np.float32)
        self.delimiter_token = self.start_token
        self.mask_token = self.start_token
        self.end_token = self.start_token

        self.initialized = True

    def _encode(self, texts, stochastic=False):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        self._lazy_init()

        batch_tokens = []
        batch_token_idxs = []
        batch_character_locs = []
        batch_char_starts = []
        for text in texts:
            sent_texts = []
            sent_vecs = []
            sent_starts = []
            sent_ends = []
            for sent in self.nlp(text).sents:
                sent_texts.append(sent.text)
                sent_vecs.append(sent.vector)
                sent_starts.append(sent.start_char)
                sent_ends.append(sent.end_char)
            batch_tokens.append(sent_texts)
            batch_token_idxs.append(sent_vecs)
            batch_character_locs.append(sent_ends)
            batch_char_starts.append(sent_starts)
        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            token_ends=batch_character_locs,
            token_starts=batch_char_starts,
        )

    def decode(self, token_ids):
        raise NotImplemented()
        
    def __getstate__(self):
        return {"Encoder": None}

    def __setstate__(self, state):
        self.__init__()
        self._lazy_init()

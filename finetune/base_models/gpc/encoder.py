import os
import json
import regex as re
from functools import lru_cache

import numpy as np

import finetune
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput, get_pairs
import sentencepiece as spm

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, 'model', 'gpc', 'encoder')
WEIRD_SPM_CHAR = "▁"


def train_tokenizer(filename, vocab_size=128000):
    spm.SentencePieceTrainer.train('--input={} --model_prefix={} --user_defined_symbols=<_start_>,<_delimiter_>,<_classify_> --unk_id=0 --vocab_size={} --input_sentence_size=10000000 --shuffle_input_sentence=true --max_sentence_length=10000000 --character_coverage=0.9999'.format(filename, ENCODER_PATH, vocab_size))


class GPCEncoder(BaseEncoder):
    UNK_IDX = 0

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=None):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

    def _lazy_init(self, errors='replace'):
        if self.initialized:
            return

        self.encoder = spm.SentencePieceProcessor()
        self.encoder.Load(ENCODER_PATH + ".model")

        self.start = self.encoder.piece_to_id('<_start_>')
        self.delimiter = self.encoder.piece_to_id('<_delimiter_>')
        self.clf_token = self.encoder.piece_to_id('<_classify_>')
        self.cache = {}

        self.initialized = True

    def _encode(self, texts, labels=None, stochastic=None):
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

            subtokens = []
            subtoken_idxs = []
            tok_pos = []
            token_start = 0
            if stochastic:
                encoded = self.encoder.sample_encode_as_pieces(text, -1, 0.1)
            else:
                encoded = self.encoder.encode_as_pieces(text)

            for j, token in enumerate(encoded):
                subtokens.append(token)
                subtoken_idxs.append(self.encoder.piece_to_id(token))
                raw_text = token.replace(WEIRD_SPM_CHAR, "")
                tok_pos.append(token_start + len(raw_text))
                token_start += len(raw_text)

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

    def decode(self, token_ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """
        return self.encoder.decode_ids(token_ids)

    def __getstate__(self):
        return {"Encoder": None}

    def __setstate__(self, state):
        self.__init__()
        self._lazy_init()

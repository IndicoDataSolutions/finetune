import os
import json
import regex as re
import logging
from functools import lru_cache

import numpy as np

import finetune
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput, get_pairs
import sentencepiece as spm

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, 'model', 'oscar', 'encoder')
WEIRD_SPM_CHAR = "▁"

LOGGER = logging.getLogger("finetune")

def train_tokenizer(filename, vocab_size=128000):
    spm.SentencePieceTrainer.train(
        (
            "--input={} --model_prefix={} --user_defined_symbols=<_start_>,<_delimiter_>,<_classify_> --unk_id=0 "
            "--vocab_size={} --input_sentence_size=10000000 --shuffle_input_sentence=true"
            " --max_sentence_length=10000000 --character_coverage=0.9999"
        ).format(filename, ENCODER_PATH, vocab_size))


class GPCEncoder(BaseEncoder):
    UNK_IDX = 0

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=None):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

    def _lazy_init(self):
        if self.initialized:
            return

        self.encoder = spm.SentencePieceProcessor()
        self.encoder.Load(ENCODER_PATH + ".model")

        self.start_token = self.encoder.piece_to_id('<_start_>')
        self.delimiter_token = self.encoder.piece_to_id('<_delimiter_>')
        self.end_token = self.encoder.piece_to_id('<_classify_>')
        self.cache = {}

        self.initialized = True

    def _encode(self, texts, labels=None, stochastic=False):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        self._lazy_init()

        batch_tokens = []
        batch_token_idxs = []
        batch_label_idxs = []
        batch_character_locs = []
        batch_char_starts = []
        label = None

        for i, text in enumerate(texts):
            text = text.replace(WEIRD_SPM_CHAR, "_")
            if labels is not None:
                label = labels[i]

            subtokens = []
            subtoken_idxs = []
            tok_pos = []
            char_starts = []
            token_start = 0
            if stochastic:
                encoded = self.encoder.sample_encode_as_pieces(text, -1, 0.1)
            else:
                encoded = self.encoder.encode_as_pieces(text)

            for j, token in enumerate(encoded):
                subtokens.append(token)
                subtoken_idxs.append(self.encoder.piece_to_id(token))
                raw_text = token.replace(WEIRD_SPM_CHAR, "")
                token_start_temp = text.find(raw_text, token_start)
                if token_start_temp == -1:
                    LOGGER.warning("SentencePiece produced a token {} not found in the original string {}".format(raw_text, text))
                else:
                    token_start = token_start_temp
                tok_pos.append(token_start + len(raw_text))
                char_starts.append(token_start)
                token_start += len(raw_text)

            batch_tokens.append(subtokens)
            batch_token_idxs.append(subtoken_idxs)
            batch_character_locs.append(tok_pos)
            batch_char_starts.append(char_starts)
            if labels is not None:
                batch_label_idxs.append([label] * len(subtoken_idxs))

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            char_locs=batch_character_locs,
            char_starts=batch_char_starts,
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

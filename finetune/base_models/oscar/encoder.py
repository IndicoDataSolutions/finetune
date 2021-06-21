import os
import json
import regex as re
import logging
from functools import lru_cache

import numpy as np

import finetune
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput, get_pairs
from finetune.util.tokenization import normalize_nfkc, WEIRD_SPM_CHAR
import sentencepiece as spm
import unicodedata

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, "model", "oscar", "encoder")

LOGGER = logging.getLogger("finetune")


def train_tokenizer(filename, vocab_size=128000):
    spm.SentencePieceTrainer.train(
        (
            "--input={} --model_prefix={} --user_defined_symbols=<_start_>,<_delimiter_>,<_classify_> --unk_id=0 "
            "--vocab_size={} --input_sentence_size=10000000 --shuffle_input_sentence=true"
            " --max_sentence_length=10000000 --character_coverage=0.9999"
        ).format(filename, ENCODER_PATH, vocab_size)
    )


class GPCEncoder(BaseEncoder):
    UNK_IDX = 0

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=None):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

        self.encoder = spm.SentencePieceProcessor()
        self.encoder.Load(ENCODER_PATH + ".model")

        self.start_token = self.encoder.piece_to_id("<_start_>")
        self.delimiter_token = self.encoder.piece_to_id("<_delimiter_>")
        self.end_token = self.encoder.piece_to_id("<_classify_>")
        self.cache = {}

    def nfck_norm_aligned(self, text):
        return normalize_nfkc(text)

    def _encode(self, texts, stochastic=False):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        batch_tokens = []
        batch_token_idxs = []
        batch_character_locs = []
        batch_char_starts = []

        for i, text in enumerate(texts):
            alignment, normed_text = self.nfck_norm_aligned(text)
            text = text.replace(WEIRD_SPM_CHAR, "_")
            subtokens = []
            subtoken_idxs = []
            tok_pos = []
            char_starts = []
            token_start = 0
            token_end = 0
            if stochastic:
                encoded = self.encoder.sample_encode_as_pieces(text, -1, 0.1)
            else:
                encoded = self.encoder.encode_as_pieces(text)

            for j, token in enumerate(encoded):
                subtoken_idxs.append(self.encoder.piece_to_id(token))
                raw_text = token.replace(WEIRD_SPM_CHAR, "")

                token_start_temp = normed_text.find(raw_text, token_end)
                if token_start_temp == -1:
                    LOGGER.warning(
                        "SentencePiece produced a token {} not found in the original string {}".format(
                            raw_text, text
                        )
                    )
                else:
                    token_start = token_start_temp
                    token_end = token_start + len(raw_text)
                real_end = alignment[token_end]
                real_start = alignment[token_start]
                subtokens.append(token)

                tok_pos.append(alignment[token_end])
                # start after the end of the last token.
                # This causes chars that become multiple tokens to get reasonable char idxs
                char_starts.append(max(alignment[token_start], tok_pos[-1]))

            batch_tokens.append(subtokens)
            batch_token_idxs.append(subtoken_idxs)
            batch_character_locs.append(tok_pos)
            batch_char_starts.append(char_starts)

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            token_ends=batch_character_locs,
            token_starts=batch_char_starts,
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

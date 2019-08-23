import spacy
import numpy as np
import os
import json

import finetune
from finetune.encoding.input_encoder import NLP, EncodedOutput, BaseEncoder, get_pairs


FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, "model", "glove", "encoder.json")


class GLOVEEncoder(BaseEncoder):

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=None):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

    def _lazy_init(self, errors="replace"):
        if self.initialized:
            return
        # Load encoder
        with open(self.encoder_path, "r") as f:
            self.encoder = json.load(f)

        self.special_tokens = ["<UNK>"]
        self.start = 0
        self.delimiter = 0
        self.clf_token = 0
        for token in self.special_tokens:
            self.encoder[token] = len(self.encoder)
        self.UNK_IDX = 0
        self.initialized = True
    
    def _encode(self, texts, labels=None, context=None):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        self._lazy_init()
        batch_tokens = []
        batch_token_idxs = []
        batch_label_idxs = []
        batch_char_ends = (
            []
        )  # to account for the fact that some BPEs have different lengths than their original tokens (e.g. special characters such as bullets)
        batch_char_starts = []
        label = None
        offset = (
            0
        )  # tracks offset between this fields' character_locs, which start at 0, and the 'start' keys in context which track the entire document (not just this field)
        skipped = 0
        for i, text in enumerate(texts):
            if labels is not None:
                label = labels[i]

            tokens = NLP(text)

            if not tokens:
                offset += len(text)  # for spans that are just whitespace
                skipped += 1
                continue
            i -= skipped
            token_texts = []
            token_idxs = []
            char_starts = []
            char_ends = []
            token_start = 0

            for j, token in enumerate(tokens):
                token_start = [token.idx]
                token_end = [token.idx + len(token)]
                token_id = [self.encoder.get(token.text, self.UNK_IDX)]
                token_idxs.extend(token_id)
                token_text = [token.text]

                token_texts.extend(token_text)
                token_char_starts = token_start + token_end[:-1]
                char_ends.extend(token_end)
                char_starts.extend(token_char_starts)

            batch_tokens.append(token_texts)
            batch_token_idxs.append(token_idxs)
            batch_char_ends.append(char_ends)
            batch_char_starts.append(char_starts)
            if labels is not None:
                batch_label_idxs.append([label] * len(token_texts))

        print(batch_label_idxs)
        print(batch_tokens)
        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            context=[],
            char_locs=batch_char_ends,
            char_starts=batch_char_starts,
        )
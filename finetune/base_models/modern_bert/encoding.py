import logging
import os
import finetune
from tokenizers import Tokenizer
from finetune.encoding.input_encoder import BaseEncoder
from finetune.encoding.input_encoder import EncodedOutput

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
TOKENIZER_PATH = os.path.join(FINETUNE_FOLDER, "model", "modern_bert", "tokenizer.json")

LOGGER = logging.getLogger("finetune")

class ModernBertEncoder(BaseEncoder):
    def __init__(self):
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        special_tokens_map = {tok.content: k for k, tok in self.tokenizer.get_added_tokens_decoder().items()}
        self.start_token = special_tokens_map["[CLS]"]
        self.delimiter_token = special_tokens_map["[SEP]"]
        self.mask_token = special_tokens_map["[MASK]"]
        self.end_token = special_tokens_map["[SEP]"]
        self.initialized = True
        self.UNK_IDX = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def _encode(self, texts):
        batch_tokens = []
        batch_token_idxs = []
        batch_char_ends = []
        batch_char_starts = []
        for text in texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            batch_tokens.append(encoded.tokens)
            batch_token_idxs.append(encoded.ids)
            token_ends = []
            token_starts = []
            for start, end in encoded.offsets:
                if token_ends:
                    # Finetune requires that tokens never overlap.
                    # This happens in huggingface tokenizers when
                    # a single character is split across multiple tokens.
                    start = max(token_ends[-1], start)
                    end = max(end, start)
                token_starts.append(start)
                token_ends.append(end)

            batch_char_ends.append(token_ends)
            batch_char_starts.append(token_starts)

        output = EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            token_ends=batch_char_ends,
            token_starts=batch_char_starts,
        )
        return output

    def decode(self, ids):
        output = self.tokenizer.decode(ids, skip_special_tokens=True)
        return output
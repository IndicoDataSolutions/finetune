import os

import finetune
from finetune.encoding.input_encoder import EncodedOutput, BaseEncoder
from finetune.base_models.bert.tokenizer import FullTokenizer

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, 'model', 'bert', 'vocab.txt')
VOCAB_PATH_MULTILINGUAL = os.path.join(FINETUNE_FOLDER, 'model', 'bert', 'vocab_multi.txt')
VOCAB_PATH_LARGE = os.path.join(FINETUNE_FOLDER, 'model', 'bert', 'vocab_large.txt')


class BERTEncoder(BaseEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """
    UNK_IDX = 0

    def __init__(self, encoder_path=None, vocab_path=VOCAB_PATH, lower_case=False):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)
        self.lower_case = lower_case

    def _lazy_init(self):
        # Must set
        if self.initialized:
            return

        self.tokenizer = FullTokenizer(vocab_file=self.vocab_path, do_lower_case=self.lower_case)

        self.start = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.delimiter = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.clf_token = self.delimiter
        self.initialized = True

    @property
    def vocab_size(self):
        self._lazy_init()
        return len(self.tokenizer.vocab)

    def _token_length(self, token):
        return len(token.strip().replace('##', ''))

    def _encode(self, texts, labels=None):
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

            subtokens, token_idxs = self.tokenizer.tokenize(text)
            subtoken_locs = [l[1] for l in token_idxs]

            batch_tokens.append(subtokens)
            batch_token_idxs.append(self.tokenizer.convert_tokens_to_ids(subtokens))
            batch_character_locs.append(subtoken_locs)
            if labels is not None:
                batch_label_idxs.append([label] * len(subtokens))

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            char_locs=batch_character_locs,
        )

    def decode(self, ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """

        raise NotImplementedError(
            "The inverse of the BERT encoder is not implemented because we have not implemented generation with BERT"
        )


class BERTEncoderLarge(BERTEncoder):

    def __init__(self, encoder_path=None, vocab_path=VOCAB_PATH_LARGE, lower_case=False):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case)


class BERTEncoderMultuilingal(BERTEncoder):

    def __init__(self, encoder_path=None, vocab_path=VOCAB_PATH_MULTILINGUAL, lower_case=False):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case)
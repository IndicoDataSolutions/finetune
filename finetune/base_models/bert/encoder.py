import os

import finetune
from finetune.encoding.input_encoder import EncodedOutput, BaseEncoder
from finetune.base_models.bert.tokenizer import FullTokenizer

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, "model", "bert", "vocab.txt")
VOCAB_PATH_MULTILINGUAL = os.path.join(
    FINETUNE_FOLDER, "model", "bert", "vocab_multi.txt"
)
VOCAB_PATH_LARGE = os.path.join(FINETUNE_FOLDER, "model", "bert", "vocab_large.txt")


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

        self.tokenizer = FullTokenizer(
            vocab_file=self.vocab_path, do_lower_case=self.lower_case
        )

        self.start = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.delimiter = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.clf_token = self.delimiter
        self.initialized = True

    @property
    def vocab_size(self):
        self._lazy_init()
        return len(self.tokenizer.vocab)

    def _token_length(self, token):
        return len(token.strip().replace("##", ""))

    def _encode(self, texts, labels=None, context=None):
        """
        Convert a batch of raw text to a batch of byte-pair encoded token indices.
        """
        self._lazy_init()
        batch_tokens = []
        batch_token_idxs = []
        batch_label_idxs = []
        batch_character_locs = []
        batch_original_character_locs = []
        batch_context = []
        label = None
        context_ = None
        offset = 0

        for i, text in enumerate(texts):
            if labels is not None:
                label = labels[i]
            if (
                context is not None
            ):  # each field in texts needs a list of dicts for its context
                context_ = context
            original_tok_pos = []

            subtokens, token_idxs, original_subtoken_positions = self.tokenizer.tokenize(
                text
            )
            subtoken_locs = [l[1] for l in token_idxs]
            original_tok_pos.extend(original_subtoken_positions)

            batch_tokens.append(subtokens)
            batch_token_idxs.append(self.tokenizer.convert_tokens_to_ids(subtokens))
            batch_character_locs.append(subtoken_locs)
            batch_original_character_locs.append(original_tok_pos)
            if labels is not None:
                batch_label_idxs.append([label] * len(subtokens))

            # Context is tokenwise, so we need to duplicate contexts for each subtoken of a token, and to match length of labels
            if context_ is not None:
                original_tokens = []
                for char_loc, token in zip(
                    batch_original_character_locs[i], batch_tokens[i]
                ):
                    original_token = 0
                    for subtoken_idx in range(len(context_)):
                        if (
                            char_loc + offset > context_[subtoken_idx]["end"]
                        ):  # subtract one since subtokens include spaces at the beginning, while the 'start's from context do not
                            original_token += 1
                    original_tokens.append(original_token)
                expanded_context = [None] * len(original_tokens)
                for j in range(len(expanded_context)):
                    expanded_context[j] = context_[original_tokens[j]]
                batch_context.append(expanded_context)
                assert len(expanded_context) == len(subtoken_locs)
                offset += batch_original_character_locs[i][-1]

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            context=batch_context,
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
    def __init__(
        self, encoder_path=None, vocab_path=VOCAB_PATH_LARGE, lower_case=False
    ):
        super().__init__(
            encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case
        )


class BERTEncoderMultuilingal(BERTEncoder):
    def __init__(
        self, encoder_path=None, vocab_path=VOCAB_PATH_MULTILINGUAL, lower_case=False
    ):
        super().__init__(
            encoder_path=encoder_path, vocab_path=vocab_path, lower_case=lower_case
        )


import os
from pathlib import Path

from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab

from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput
from finetune.util.download import FINETUNE_BASE_FOLDER


class GloveEncoder(BaseEncoder):

    # Does not include special tokens
    N_VOCAB = 1070971 

    def __init__(self):
        # encoder / vocab managed by spacy
        super().__init__(encoder_path=None, vocab_path=None)

    def _lazy_init(self):
        if self.initialized:
            return
        
        # TODO: can't figure out how to prevent vectors from being loaded.
        # Maybe use tokenizer.from_disk?
        # self._tokenizer = Tokenizer.from_disk()
        tokenizer_path = os.path.join(FINETUNE_BASE_FOLDER, 'model', 'glove', 'tokenizer.spacy')
        vocab_path = os.path.join(FINETUNE_BASE_FOLDER, 'model', 'glove', 'vocab.spacy')

        self.vocab = Vocab().from_disk(Path(vocab_path))
        self.tokenizer = Tokenizer(self.vocab).from_disk(Path(tokenizer_path))
        self._add_extra_tokens() 
        self.initialized = True

    def _add_extra_tokens(self):
        self.special_tokens = ['_start_', '_delimiter_', '_classify_', '_oov_']
        self.start_token = self.N_VOCAB 
        self.delimiter_token = self.N_VOCAB + 1
        self.end_token = self.N_VOCAB + 2
        self.oov_token = self.N_VOCAB + 3
        self.clf_token = self.end_token
 
    @property
    def vocab_size(self):
        return self.N_VOCAB + len(self.special_tokens)

    def _token_length(self, token):
        return len(token)

    def _encode(self, texts):
        # TODO -- take advantage of pipelining
        self._lazy_init()
        docs = self.tokenizer.pipe(texts)

        batch_tokens = []
        batch_token_idxs = []
        batch_char_starts = []
        batch_char_ends = []
        
        for i, doc in enumerate(docs):
            batch_tokens.append([token.text for token in doc])
            batch_token_idxs.append([
                self.tokenizer.vocab.vectors.key2row[token.orth] 
                if token.has_vector else self.oov_token
                for token in doc
            ])
            batch_char_starts.append([token.idx for token in doc])
            batch_char_ends.append([token.idx + len(token.text) for token in doc])
 
        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            token_ends=batch_char_ends,
            token_starts=batch_char_starts
        )

    def decode(self, token_ids):
        raise NotImplemented("Because spacy's Glove implementation uses a hashing vectorizer, we cannot deterministically decode")

import os
import json
import regex as re
from functools import lru_cache

import numpy as np

import finetune
from finetune.encoding.input_encoder import BaseEncoder, EncodedOutput, get_pairs

FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
ENCODER_PATH = os.path.join(FINETUNE_FOLDER, 'model', 'gpt2', 'encoder.json')
VOCAB_PATH = os.path.join(FINETUNE_FOLDER, 'model', 'gpt2', 'vocab.bpe')

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class GPT2Encoder(BaseEncoder):
    """
    A modified wrapper for a public python BPE tokenizer. The modifications allow encoding directly into the formats
    required for finetune. Particularly with respect to formatting with multiple inputs.
    """
    UNK_IDX = 0

    def __init__(self, encoder_path=ENCODER_PATH, vocab_path=VOCAB_PATH):
        super().__init__(encoder_path=encoder_path, vocab_path=vocab_path)

    def _lazy_init(self, errors='replace'):
        if self.initialized:
            return

        # Load encoder
        with open(self.encoder_path, 'r') as f:
            self.encoder = json.load(f)

        # Load BPE
        with open(self.vocab_path, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.special_tokens = ['_delimiter_', '_classify_']
        for token in self.special_tokens:
            self.encoder[token] = len(self.encoder)

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.start = self.encoder['<|endoftext|>']
        self.delimiter = self.encoder['_delimiter_']
        self.clf_token = self.encoder['_classify_']
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.initialized = True

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def _encode(self, texts, labels=None, context=None):
        """
        Convert a sample of raw text to a list of byte-pair encoded token indices.
        """
        self._lazy_init()

        batch_tokens = []
        batch_token_idxs = []
        batch_label_idxs = []
        batch_context_idxs = []
        batch_character_locs = []
        batch_context = []
        label = None
        context_ = None


        #print("BEFORE ENCODE")
        #print(context)
        for i, text in enumerate(texts):
            #print("WITHIN LOOP")
            #print(i)
            #print(text)
            if labels is not None:
                label = labels[i]
            if context is not None:
                context_ = context[i]
                if type(context_) == dict: # seeing if we have a list of context tokens, or a list of lists of context tokens
                    context_ = context
            #print(context_)

            subtokens = []
            subtoken_idxs = []
            tok_pos = []
            token_start = 0

            tokens = re.findall(self.pat, text)
            for j, token in enumerate(tokens):
                encoded_token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                bpe_toks = self.bpe(encoded_token).split(' ')
                try:
                    if token.strip():
                        token_start = text.index(token, token_start)
                except ValueError:
                    # text_standardization oddity
                    traceback.print_exc()
                    continue

                subtokens.extend(bpe_toks)
                subtoken_idxs.extend([
                    self.encoder.get(t, self.UNK_IDX)
                    for t in bpe_toks
                ])
                subtoken_positions = np.cumsum([len(tok) for tok in bpe_toks]) + token_start
                token_start += len(token.strip())
                tok_pos.extend(subtoken_positions)

            batch_tokens.append(subtokens)
            batch_token_idxs.append(subtoken_idxs)
            batch_character_locs.append(tok_pos)
            if labels is not None:
                batch_label_idxs.append([label] * len(subtoken_idxs))
        
            # Context is tokenwise, so we need to duplicate contexts for each subtoken of a token, and to match length of labels
            if context_ is not None:
                single_example=False
                #print("CONTEXT_ WITHIN ENCODE")
                #print(context_)
                #print(type(context_))
                try:
                    token_starts = [context['start'] for context in context_]
                except TypeError: # only one token of context, so the start must be 0 TODO: find out why this throws such an error
                    token_starts = [0]
                    1/0
                    single_example=True
                original_tokens = []
                for char_loc in batch_character_locs[i]:
                    original_token=-1
                    for i in range(len(token_starts)):
                        if char_loc >= token_starts[i]:
                            original_token += 1
                    original_tokens.append(original_token)
                expanded_context = [None]*len(original_tokens)
                for j in range(len(expanded_context)):
                    expanded_context[j] =  context_ if single_example else context_[original_tokens[j]]
                batch_context.append(expanded_context)

                assert len(expanded_context) == len(subtoken_idxs) and len(expanded_context) == len(tok_pos)
                
        #print("AFTER ENCODE")
                    
        #print(batch_context)

        return EncodedOutput(
            token_ids=batch_token_idxs,
            tokens=batch_tokens,
            labels=batch_label_idxs,
            context=batch_context,
            char_locs=batch_character_locs
        )

    def decode(self, token_ids):
        """
        Convert a batch of ids [batch_size, id] into text(ish).
        """
        text = ''.join([self.decoder[token_id] for token_id in token_ids])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

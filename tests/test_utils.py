import unittest
import os.path
import random
import json
from collections import Counter
import math

import numpy as np
import tensorflow as tf
import pandas as pd
import joblib as jl

import unicodedata

import finetune
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.optimizers.gradient_accumulation import get_grad_accumulation_optimizer
from finetune.util.imbalance import compute_class_weights
from finetune.util.timing import ProgressBar
from finetune.errors import FinetuneError
from finetune import Classifier, SequenceLabeler
from finetune.base_models import GPT, GPT2, BERT
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoderV2, RoBERTaEncoder
from finetune.base_models.bert.encoder import BERTEncoderMultuilingal, BERTEncoder
from finetune.base_models.oscar.encoder import GPCEncoder

class TestGPTEncoder(unittest.TestCase):
    Encoder = GPTEncoder

    def setUp(self):
        self.encoder = self.Encoder()
        self.text = """This is a basic test of tokenizers but with some fun things at the end. 
This'll "tejhirpwkjovf[-d" the encoder's abilities to deal with ½ a sentence of rubbish… and some ellipsis...
Some Emojis 🐕\t 🦄. 田中さんにあげて下さいパーティーへ行かないか和製漢語部落格사회과학원 어학연구소찦차를 타고 온 펲시맨과 쑛다리 똠방각하
社會科學院語學研究所
울란바토르
𠜎𠜱𠝹𠱓𠱸𠲖𠳏
Ｔｈｅ ｑｕｉｃｋ ｂｒｏｗｎ ｆｏｘ ｊｕｍｐｓ ｏｖｅｒ ｔｈｅ ｌａｚｙ ｄｏｇ
𝐓𝐡𝐞 𝐪𝐮𝐢𝐜𝐤 𝐛𝐫𝐨𝐰𝐧 𝐟𝐨𝐱 𝐣𝐮𝐦𝐩𝐬 𝐨𝐯𝐞𝐫 𝐭𝐡𝐞 𝐥𝐚𝐳𝐲 𝐝𝐨𝐠
𝕿𝖍𝖊 𝖖𝖚𝖎𝖈𝖐 𝖇𝖗𝖔𝖜𝖓 𝖋𝖔𝖝 𝖏𝖚𝖒𝖕𝖘 𝖔𝖛𝖊𝖗 𝖙𝖍𝖊 𝖑𝖆𝖟𝖞 𝖉𝖔𝖌
𝑻𝒉𝒆 𝒒𝒖𝒊𝒄𝒌 𝒃𝒓𝒐𝒘𝒏 𝒇𝒐𝒙 𝒋𝒖𝒎𝒑𝒔 𝒐𝒗𝒆𝒓 𝒕𝒉𝒆 𝒍𝒂𝒛𝒚 𝒅𝒐𝒈
𝓣𝓱𝓮 𝓺𝓾𝓲𝓬𝓴 𝓫𝓻𝓸𝔀𝓷 𝓯𝓸𝔁 𝓳𝓾𝓶𝓹𝓼 𝓸𝓿𝓮𝓻 𝓽𝓱𝓮 𝓵𝓪𝔃𝔂 𝓭𝓸𝓰
𝕋𝕙𝕖 𝕢𝕦𝕚𝕔𝕜 𝕓𝕣𝕠𝕨𝕟 𝕗𝕠𝕩 𝕛𝕦𝕞𝕡𝕤 𝕠𝕧𝕖𝕣 𝕥𝕙𝕖 𝕝𝕒𝕫𝕪 𝕕𝕠𝕘
𝚃𝚑𝚎 𝚚𝚞𝚒𝚌𝚔 𝚋𝚛𝚘𝚠𝚗 𝚏𝚘𝚡 𝚓𝚞𝚖𝚙𝚜 𝚘𝚟𝚎𝚛 𝚝𝚑𝚎 𝚕𝚊𝚣𝚢 𝚍𝚘𝚐
⒯⒣⒠ ⒬⒰⒤⒞⒦ ⒝⒭⒪⒲⒩ ⒡⒪⒳ ⒥⒰⒨⒫⒮ ⒪⒱⒠⒭ ⒯⒣⒠ ⒧⒜⒵⒴ ⒟⒪⒢
<script src="/\%(jscript)s"></script>
<IMG SRC=&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;&#97;&#108;&#101;&#114;&#116;&#40;&#39;&#88;&#83;&#83;&#39;&#41;>
<IMG SRC=&#0000106&#0000097&#0000118&#0000097&#0000115&#0000099&#0000114&#0000105&#0000112&#0000116&#0000058&#0000097&#0000108&#0000101&#0000114&#0000116&#0000040&#0000039&#0000088&#0000083&#0000083&#0000039&#0000041>
<IMG SRC=&#x6A&#x61&#x76&#x61&#x73&#x63&#x72&#x69&#x70&#x74&#x3A&#x61&#x6C&#x65&#x72&#x74&#x28&#x27&#x58&#x53&#x53&#x27&#x29>
<IMG SRC="jav&#x0D;ascript:alert('XSS');">
Powerلُلُصّبُلُلصّبُررً ॣ ॣh ॣ ॣ冗
🏳0🌈️
జ్ఞ‌ా
گچپژ
❑
"""

    def test_max_length(self):
        encoded = self.encoder.encode_multi_input([self.text], max_length=20)
        self.assertEqual(len(encoded.tokens), 20)

    def test_empty_string(self):
        # This test is important for cached predict.
        encoded = self.encoder.encode_multi_input([""], max_length=20)
        self.assertEqual(len(encoded.tokens), 2) # start and end tokens

    def test_no_whitespace_in_idxs(self):
        def make_comparible(text):
            return unicodedata.normalize("NFKC", text.replace("\u200c", "")).lower()
        encoded = self.encoder.encode_multi_input([self.text], max_length=2000)
        print(encoded)
        for tok, start, end in zip(encoded.tokens, encoded.token_starts, encoded.token_ends):
            if start == -1:
                continue # this is the special tokens
            print(start, end)
            sub_seq = self.text[start: end]
            self.assertEqual(sub_seq, sub_seq.strip()) # no leading or trailing whitespace
            self.assertNotIn("\n", sub_seq)
            self.assertIn(make_comparible(sub_seq), make_comparible(tok))

    def test_end_alignment(self):
        encoded = self.encoder.encode_multi_input([self.text], max_length=2000)
        self.assertEqual(encoded.token_ends[-2], len(self.text.rstrip()))

class TestGPT2Encoder(TestGPTEncoder):
    Encoder = GPT2Encoder

class TestRobertaEncoder(TestGPTEncoder):
    Encoder = RoBERTaEncoder

class TestRobertaV2Encoder(TestGPTEncoder):
    Encoder = RoBERTaEncoderV2

class TestBertEncoderMulti(TestGPTEncoder):
    Encoder = BERTEncoderMultuilingal
    
class TestBertEncoder(TestGPTEncoder):
    Encoder = BERTEncoder

class TestOscarEncoder(TestGPTEncoder):
    Encoder = GPCEncoder
    
class TestFinetuneIndicoConverters(unittest.TestCase):

    def test_invalid_keyword(self):
        with self.assertRaises(FinetuneError):
            model = Classifier(tensorboard='./testing') # should be tensorboard_folder
    

    def test_whitespace_handling(self):
        # Newline complications
        finetunex = [["Train:", "\n\n\n and test", " tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]
        expectedx = ["Train:\n\n\n and test tokenization must be equivalent"]
        expectedy = [
            [
                {'start': 0, 'end': 18, 'label': "1", 'text': "Train:\n\n\n and test"},
                {'start': 10, 'end': 39, 'label': "2", 'text': "and test tokenization must be"}
            ]
        ]
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)
    
        expectedx = ["Train and test tokenization must be equivalent"]
        expectedy = [
            [
                {'start': 0, 'end': 14, 'label': "1", 'text': "Train and test"},
                {'start': 6, 'end': 35, 'label': "2", 'text': "and test tokenization must be"}
            ]
        ]
    
        # Spaces before labels
        finetunex = [["Train", " and test", " tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]

        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

        # Spaces after labels
        finetunex = [["Train ", "and test ", "tokenization must be ", "equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]
    
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

        # Whitespace anarchy
        finetunex = [["Train", " and test ", "tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]

        indicox_pred, indicoy_pred = finetune_to_indico_sequence(expectedx, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False)
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)
        

    def test_overlapping(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico", "Is the", "best", "hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2",), ("<PAD>",)]
        ]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>",
                                                                 subtoken_predictions=True)

        indicoy = [
            [
                {'start': 0, 'end': 13, 'label': '1', 'text': 'Indico Is the'},
                {'start': 7, 'end': 18, 'label': '2', 'text': 'Is the best'},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_overlapping_gpt2(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico", " Is the", " best", " hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2", ), ("<PAD>")]
        ]
        encoder = GPT2Encoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>",
                                                                 subtoken_predictions=False)
        indicoy = [
            [
                {'start': 0, 'end': 13, 'label': '1', 'text': 'Indico Is the'},
                {'start': 7, 'end': 18, 'label': '2', 'text': 'Is the best'},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_overlapping_gpt2_subtokens(self):
        raw = ["Indico Is the best hey"]
        finetunex = [
            ["Indico", " Is the", " best", " hey"]
        ]
        finetuney = [
            [("1",), ("1", "2"), ("2", ), ("<PAD>")]
        ]
        encoder = GPT2Encoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>",
                                                                 subtoken_predictions=True)

        indicoy = [
            [
                {'start': 0, 'end': 13, 'label': '1', 'text': 'Indico Is the'},
                {'start': 6, 'end': 18, 'label': '2', 'text': ' Is the best'},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_nested_labels(self):
        raw = ["Indico Is the best"]
        finetunex = [
            ["Indico ", "Is the", " best"]
        ]
        finetuney = [
            [("1", ), ("1", "2", "3"), ("1", )]
        ]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>")


    def test_three_overlapping_labels(self):
        raw = ["Indico Is the very best"]
        finetunex = [
            ["Indico ", "Is the very", " best"]
        ]
        finetuney = [
            [("<PAD>", ), ("1", "2", "3"), ("1", "3")]
        ]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(raw, finetunex, finetuney, none_value="<PAD>")
        indicoy_pred = [sorted(seq, key=lambda x: x['label']) for seq in indicoy_pred]
        indicoy = [
            sorted(
                [
                    {'start': 7, 'end': 18, 'label': '2', 'text': 'Is the very'},
                    {'start': 7, 'end': 23, 'label': '1', 'text': 'Is the very best'},
                    {'start': 7, 'end': 23, 'label': '3', 'text': 'Is the very best'}
                ],
                key=lambda x: x['label']
            )
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_compute_class_weights(self):
        # regression test for issue #181
        np.random.seed(0)
        y = np.random.choice(a=[0, 1, 2], size=1000, p=[0.3, 0.6, 0.1])
        class_counts = Counter(y)
        weights = compute_class_weights('log', class_counts=class_counts)
        self.assertEqual(weights[1], 1.0)


class TestGradientAccumulation(unittest.TestCase):

    @tf.function
    def test_gradient_accumulating_optimizer(self):
        with tf.Graph().as_default():
            loss = tf.compat.v1.get_variable("loss", shape=1)
            lr = 0.1
            opt = get_grad_accumulation_optimizer(tf.keras.optimizers.SGD, 2)(lr)
            global_step = tf.compat.v1.train.get_or_create_global_step()
            with tf.control_dependencies([opt.minimize(lambda: tf.abs(loss), [loss])]):
                train_op = global_step.assign_add(1)

            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(100):
                val_before = sess.run(loss)
                grad_before = np.sign(val_before)
                sess.run(train_op)

                val_after1 = sess.run(loss)
                grad_after1 = np.sign(val_after1)
                sess.run(train_op)

                val_after2 = sess.run(loss)

                self.assertEqual(val_before - (grad_before + grad_after1) * lr, val_after2)
                self.assertEqual(val_before, val_after1)  # first step should not actually do anything


class TestProgressBar(unittest.TestCase):

    def test_progress_bar(self):
        state = {'hook_run': False}
    
        def update_state(timing_dict):
            nonlocal state
            state['hook_run'] = True

        pbar = ProgressBar(range(1000), update_hook=update_state)
        assert state['hook_run']

if __name__ == '__main__':
    unittest.main()

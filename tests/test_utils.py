import unittest
from collections import Counter
import pytest

import numpy as np
import tensorflow as tf

import unicodedata

from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.optimizers.gradient_accumulation import get_grad_accumulation_optimizer
from finetune.util.imbalance import compute_class_weights
from finetune.util.optimize_loss import OPTIMIZERS
from finetune.util.timing import ProgressBar
from finetune.errors import FinetuneError
from finetune import Classifier, SequenceLabeler
from finetune.base_models.gpt.encoder import GPTEncoder
from finetune.base_models.gpt2.encoder import GPT2Encoder
from finetune.base_models.bert.roberta_encoder import (
    RoBERTaEncoderV2,
    RoBERTaEncoder,
    RoBERTaEncoderSlow,
)
from finetune.base_models.bert.encoder import BERTEncoderMultuilingal, BERTEncoder
from finetune.base_models.oscar.encoder import GPCEncoder
from finetune.scheduler import Scheduler


class TestGPTEncoder(unittest.TestCase):
    Encoder = GPTEncoder

    def setUp(self):
        self.encoder = self.Encoder()
        with open("tests/data/weird_text.txt") as f:
            weird_text = "".join(f.readlines()).rstrip()
        self.text = weird_text

    def test_max_length(self):
        encoded = self.encoder.encode_multi_input([self.text], max_length=20)
        self.assertEqual(len(encoded.tokens), 20)

    def test_empty_string(self):
        # This test is important for cached predict.
        encoded = self.encoder.encode_multi_input([""], max_length=20)
        self.assertEqual(len(encoded.tokens), 2)  # start and end tokens

    def test_no_whitespace_in_idxs(self):
        def make_comparible(text):
            return unicodedata.normalize("NFKC", text.replace("\u200c", "")).lower()

        encoded = self.encoder.encode_multi_input([self.text], max_length=2000)
        print(encoded)
        for tok, start, end in zip(
            encoded.tokens, encoded.token_starts, encoded.token_ends
        ):
            if start == -1:
                continue  # this is the special tokens
            print(start, end)
            sub_seq = self.text[start:end]
            self.assertEqual(
                sub_seq, sub_seq.strip()
            )  # no leading or trailing whitespace
            self.assertNotIn("\n", sub_seq)
            self.assertIn(make_comparible(sub_seq), make_comparible(tok))

    def test_end_alignment(self):
        encoded = self.encoder.encode_multi_input([self.text], max_length=2000)
        self.assertEqual(encoded.token_ends[-2], len(self.text))


class TestGPT2Encoder(TestGPTEncoder):
    Encoder = GPT2Encoder


class TestRobertaEncoder(TestGPTEncoder):
    Encoder = RoBERTaEncoder


class TestRobertaEncoderSlow(TestGPTEncoder):
    Encoder = RoBERTaEncoderSlow


class TestRobertaV2Encoder(TestGPTEncoder):
    Encoder = RoBERTaEncoderV2

    @pytest.mark.xfail
    def test_no_whitespace_in_idxs(self):
        super().test_no_whitespace_in_idxs()


class TestBertEncoderMulti(TestGPTEncoder):
    Encoder = BERTEncoderMultuilingal


class TestBertEncoder(TestGPTEncoder):
    Encoder = BERTEncoder


class TestOscarEncoder(TestGPTEncoder):
    Encoder = GPCEncoder


class TestFinetuneIndicoConverters(unittest.TestCase):
    def test_invalid_keyword(self):
        with self.assertRaises(FinetuneError):
            model = Classifier(tensorboard="./testing")  # should be tensorboard_folder

    def test_whitespace_handling(self):
        # Newline complications
        finetunex = [
            ["Train:", "\n\n\n and test", " tokenization must be", " equivalent"]
        ]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]
        expectedx = ["Train:\n\n\n and test tokenization must be equivalent"]
        expectedy = [
            [
                {"start": 0, "end": 18, "label": "1", "text": "Train:\n\n\n and test"},
                {
                    "start": 10,
                    "end": 39,
                    "label": "2",
                    "text": "and test tokenization must be",
                },
            ]
        ]
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            expectedx,
            finetunex,
            finetuney,
            none_value="<PAD>",
            subtoken_predictions=False,
        )
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

        expectedx = ["Train and test tokenization must be equivalent"]
        expectedy = [
            [
                {"start": 0, "end": 14, "label": "1", "text": "Train and test"},
                {
                    "start": 6,
                    "end": 35,
                    "label": "2",
                    "text": "and test tokenization must be",
                },
            ]
        ]

        # Spaces before labels
        finetunex = [["Train", " and test", " tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]

        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            expectedx,
            finetunex,
            finetuney,
            none_value="<PAD>",
            subtoken_predictions=False,
        )
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

        # Spaces after labels
        finetunex = [["Train ", "and test ", "tokenization must be ", "equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]

        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            expectedx,
            finetunex,
            finetuney,
            none_value="<PAD>",
            subtoken_predictions=False,
        )
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

        # Whitespace anarchy
        finetunex = [["Train", " and test ", "tokenization must be", " equivalent"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]

        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            expectedx,
            finetunex,
            finetuney,
            none_value="<PAD>",
            subtoken_predictions=False,
        )
        self.assertEqual(indicox_pred, expectedx)
        self.assertEqual(indicoy_pred, expectedy)

    def test_overlapping(self):
        raw = ["Indico Is the best hey"]
        finetunex = [["Indico", "Is the", "best", "hey"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>",)]]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=True
        )

        indicoy = [
            [
                {"start": 0, "end": 13, "label": "1", "text": "Indico Is the"},
                {"start": 7, "end": 18, "label": "2", "text": "Is the best"},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_overlapping_gpt2(self):
        raw = ["Indico Is the best hey"]
        finetunex = [["Indico", " Is the", " best", " hey"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>")]]
        encoder = GPT2Encoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=False
        )
        indicoy = [
            [
                {"start": 0, "end": 13, "label": "1", "text": "Indico Is the"},
                {"start": 7, "end": 18, "label": "2", "text": "Is the best"},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_overlapping_gpt2_subtokens(self):
        raw = ["Indico Is the best hey"]
        finetunex = [["Indico", " Is the", " best", " hey"]]
        finetuney = [[("1",), ("1", "2"), ("2",), ("<PAD>")]]
        encoder = GPT2Encoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, none_value="<PAD>", subtoken_predictions=True
        )

        indicoy = [
            [
                {"start": 0, "end": 13, "label": "1", "text": "Indico Is the"},
                {"start": 7, "end": 18, "label": "2", "text": "Is the best"},
            ]
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_nested_labels(self):
        raw = ["Indico Is the best"]
        finetunex = [["Indico ", "Is the", " best"]]
        finetuney = [[("1",), ("1", "2", "3"), ("1",)]]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, none_value="<PAD>"
        )

    def test_three_overlapping_labels(self):
        raw = ["Indico Is the very best"]
        finetunex = [["Indico ", "Is the very", " best"]]
        finetuney = [[("<PAD>",), ("1", "2", "3"), ("1", "3")]]
        encoder = GPTEncoder()
        indicox_pred, indicoy_pred = finetune_to_indico_sequence(
            raw, finetunex, finetuney, none_value="<PAD>"
        )
        indicoy_pred = [sorted(seq, key=lambda x: x["label"]) for seq in indicoy_pred]
        indicoy = [
            sorted(
                [
                    {"start": 7, "end": 18, "label": "2", "text": "Is the very"},
                    {"start": 7, "end": 23, "label": "1", "text": "Is the very best"},
                    {"start": 7, "end": 23, "label": "3", "text": "Is the very best"},
                ],
                key=lambda x: x["label"],
            )
        ]
        self.assertEqual(indicoy, indicoy_pred)
        self.assertEqual(raw, indicox_pred)

    def test_compute_class_weights(self):
        # regression test for issue #181
        np.random.seed(0)
        y = np.random.choice(a=[0, 1, 2], size=1000, p=[0.3, 0.6, 0.1])
        class_counts = Counter(y)
        weights = compute_class_weights("log", class_counts=class_counts)
        self.assertEqual(weights[1], 1.0)


def body_of_test_gradient_accumulating_optimizer(opt, accumulate_on_cpu):
    with tf.Graph().as_default():
        loss = tf.compat.v1.get_variable("loss", shape=1)
        lr = 0.1
        opt = get_grad_accumulation_optimizer(opt, 2, accumulate_on_cpu=accumulate_on_cpu)(lr)
        global_step = tf.compat.v1.train.get_or_create_global_step()
        if isinstance(opt, tf.keras.optimizers.Optimizer):
            with tf.control_dependencies(
                [opt.minimize(lambda: tf.abs(loss), [loss])]
            ):
                train_op = global_step.assign_add(1)
        else:
            train_op = opt.minimize(tf.abs(loss), global_step=global_step)

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

            gs = sess.run(global_step)
            assert val_before == val_after1
            assert val_before - (grad_before + grad_after1) * lr == val_after2
            assert gs == (i + 1) * 2

@pytest.mark.parametrize("accumulate_on_cpu", [True, False])
def test_gradient_accumulating_optimizer_keras(accumulate_on_cpu):
    body_of_test_gradient_accumulating_optimizer(tf.keras.optimizers.SGD, accumulate_on_cpu)

@pytest.mark.parametrize("accumulate_on_cpu", [True, False])
def test_gradient_accumulating_optimizer_compat(accumulate_on_cpu):
    body_of_test_gradient_accumulating_optimizer(
        tf.compat.v1.train.GradientDescentOptimizer,
        accumulate_on_cpu=accumulate_on_cpu
    )


class TestProgressBar(unittest.TestCase):
    def test_progress_bar(self):
        state = {"hook_run": False}

        def update_state(timing_dict):
            nonlocal state
            state["hook_run"] = True

        pbar = ProgressBar(range(1000), update_hook=update_state)
        assert state["hook_run"]


class TestOptimizers(unittest.TestCase):
    @tf.function
    def test_optimizers(self):
        for opt_class in OPTIMIZERS.values():
            with tf.Graph().as_default():
                loss_var = tf.compat.v1.get_variable("loss", shape=1)
                loss = tf.abs(loss_var)
                lr = 0.1
                opt = opt_class(lr, weight_decay=1e-10, decay_var_list=[loss_var])
                if isinstance(opt, tf.keras.optimizers.Optimizer):
                    train_op = opt.minimize(lambda: loss, [loss_var])
                else:
                    train_op = opt.minimize(loss)

                sess = tf.compat.v1.Session()
                sess.run(tf.compat.v1.global_variables_initializer())
                original_loss = sess.run(loss)
                for i in range(10):
                    sess.run(train_op)
            self.assertLess(sess.run(loss), original_loss)


class TestSaveMultiple(unittest.TestCase):
    
    def preds_equal(self, a, b):
        assert len(a) == len(b)
        for ai, bi in zip(a, b):
            assert len(ai) == len(bi)
            for aii, bii in zip(ai, bi):
                assert aii["text"] == bii["text"]
                assert aii["start"] == bii["start"]
                assert aii["end"] == bii["end"]
                assert len(aii["confidence"]) == len(bii["confidence"])
                assert aii["confidence"].keys() == bii["confidence"].keys()
                for k in aii["confidence"].keys():
                    self.assertAlmostEqual(aii["confidence"][k], bii["confidence"][k])

    def test_save_single(self):
        model = SequenceLabeler()
        model.fit(["test text"] * 20, [[{"start": 0, "end": 4, "label": "test"}]] * 20)
        preds_a = model.predict(["test text"])
        SequenceLabeler.save_multiple("multiple_models.jl", {"a": model})
        loaded_model = SequenceLabeler.load("multiple_models.jl", key="a")
        preds_b = loaded_model.predict(["test text"])
        shed = Scheduler()
        preds_c = shed.predict("multiple_models.jl", ["test text"], key="a")
        assert preds_a[0]  # If this is empty it's not a good test that nothing changed.
        self.preds_equal(preds_a, preds_b)
        self.preds_equal(preds_b, preds_c)

    def test_save_mutliple(self):
        model_a = SequenceLabeler()
        model_b = SequenceLabeler()
        model_a.fit(["test text"] * 20, [[{"start": 0, "end": 4, "label": "test"}]] * 20)
        model_b.fit(["test text"] * 20, [[{"start": 5, "end": 9, "label": "test"}]] * 20)

        preds_a_1 = model_a.predict(["test text"])
        preds_b_1 = model_b.predict(["test text"])
        
        SequenceLabeler.save_multiple("multiple_models.jl", {"a": model_a, "b": model_b})
        shed = Scheduler()
        preds_a_2 = shed.predict("multiple_models.jl", ["test text"], key="a")
        preds_b_2 = shed.predict("multiple_models.jl", ["test text"], key="b")
        self.preds_equal(preds_a_1, preds_a_2)
        self.preds_equal(preds_b_1, preds_b_2)

    
    def test_save_mutliple_inc_non_model(self):
        model_a = SequenceLabeler()
        model_a.fit(["test text"] * 20, [[{"start": 0, "end": 4, "label": "test"}]] * 20)
        preds_a_1 = model_a.predict(["test text"])
    
        SequenceLabeler.save_multiple("multiple_models.jl", {"a": model_a, "a_preds": preds_a_1})
        
        shed = Scheduler()
        preds_a_2 = shed.predict("multiple_models.jl", ["test text"], key="a")
        assert preds_a_1 == SequenceLabeler.load("multiple_models.jl", key="a_preds")
        self.preds_equal(preds_a_1, preds_a_2)


    


if __name__ == "__main__":
    unittest.main()

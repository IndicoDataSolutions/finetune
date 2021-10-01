import os
import unittest
import logging
from copy import copy, deepcopy
from pathlib import Path
import codecs
import json
import random
import time
import weakref
import gc

# required for tensorflow logging control
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest
from pytest import approx

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from finetune import SequenceLabeler
from finetune.base_models import GPT
from finetune.config import get_config
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
from finetune.util.metrics import (
    sequence_labeling_token_precision,
    sequence_labeling_token_recall,
    sequence_labeling_overlap_precision,
    sequence_labeling_overlap_recall,
)

SKIP_LM_TESTS = get_config().base_model.is_bidirectional


class TestSequenceLabeler(unittest.TestCase):

    n_sample = 100
    dataset_path = os.path.join("Data", "Sequence", "reuters.xml")
    processed_path = os.path.join("Data", "Sequence", "reuters.json")

    @classmethod
    def _download_reuters(cls):
        """
        Download Stanford Sentiment Treebank to enso `data` directory
        """
        path = Path(cls.dataset_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(cls.dataset_path):
            url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
            r = requests.get(url)
            with open(cls.dataset_path, "wb") as fp:
                fp.write(r.content)

        with codecs.open(cls.dataset_path, "r", "utf-8") as infile:
            soup = bs(infile, "html.parser")

        docs = []
        docs_labels = []
        for elem in soup.find_all("document"):
            texts = []
            labels = []

            # Loop through each child of the element under "textwithnamedentities"
            for c in elem.find("textwithnamedentities").children:
                if type(c) == Tag:
                    if c.name == "namedentityintext":
                        label = "Named Entity"  # part of a named entity
                    else:
                        label = "<PAD>"  # irrelevant word
                    texts.append(c.text)
                    labels.append(label)

            docs.append(texts)
            docs_labels.append(labels)

        with open(cls.processed_path, "wt") as fp:
            json.dump((docs, docs_labels), fp)

    @classmethod
    def setUpClass(cls):
        cls._download_reuters()

    def default_config(self, **kwargs):
        d = dict(
            base_model=GPT,
            batch_size=2,
            max_length=256,
            lm_loss_coef=0.0,
            val_size=0,
        )
        d.update(**kwargs)
        return d

    def setUp(self):
        self.save_file = "tests/saved-models/test-save-load"
        random.seed(42)
        np.random.seed(42)
        with open(self.processed_path, "rt") as fp:
            self.texts, self.labels = json.load(fp)

        self.model = SequenceLabeler(**self.default_config())

    @pytest.mark.skipif(
        SKIP_LM_TESTS, reason="Bidirectional models do not yet support LM functions"
    )
    def test_fit_lm_only(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1
        )
        self.model.fit(train_texts)
        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)
        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]["confidence"], dict)
        token_precision = sequence_labeling_token_precision(
            test_annotations, predictions
        )
        token_recall = sequence_labeling_token_recall(test_annotations, predictions)
        overlap_precision = sequence_labeling_overlap_precision(
            test_annotations, predictions
        )
        overlap_recall = sequence_labeling_overlap_recall(test_annotations, predictions)
        self.assertIn("Named Entity", token_precision)
        self.assertIn("Named Entity", token_recall)
        self.assertIn("Named Entity", overlap_precision)
        self.assertIn("Named Entity", overlap_recall)
        self.model.save(self.save_file)
        model = SequenceLabeler.load(self.save_file)
        predictions = model.predict(test_texts)

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        Ensure class reweighting behaves as intended
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1, random_state=42
        )

        reweighted_model = SequenceLabeler(
            **self.default_config(class_weights={"Named Entity": 10.0})
        )
        reweighted_model.fit(train_texts, train_annotations)
        reweighted_predictions = reweighted_model.predict(test_texts)
        reweighted_token_recall = sequence_labeling_token_recall(
            test_annotations, reweighted_predictions
        )

        self.model.fit(train_texts, train_annotations)
        predictions = self.model.predict(test_texts)
        per_token_predictions = self.model.predict(test_texts, per_token=True)
        with_doc_probas = self.model.predict(
            test_texts, return_negative_confidence=True
        )
        probas = self.model.predict_proba(test_texts)

        for pred, pred_with_prob in zip(predictions, with_doc_probas):
            self.assertEqual(pred, pred_with_prob["prediction"])
            self.assertIsInstance(pred_with_prob["negative_confidence"], dict)

        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]["confidence"], dict)

        token_precision = sequence_labeling_token_precision(
            test_annotations, predictions
        )
        token_recall = sequence_labeling_token_recall(test_annotations, predictions)
        overlap_precision = sequence_labeling_overlap_precision(
            test_annotations, predictions
        )
        overlap_recall = sequence_labeling_overlap_recall(test_annotations, predictions)

        self.assertIn("Named Entity", token_precision)
        self.assertIn("Named Entity", token_recall)
        self.assertIn("Named Entity", overlap_precision)
        self.assertIn("Named Entity", overlap_recall)

        self.model.save(self.save_file)

        self.assertGreater(
            reweighted_token_recall["Named Entity"], token_recall["Named Entity"]
        )

    def test_cached_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, _ = train_test_split(
            texts, annotations, test_size=0.1
        )
        self.model.fit(train_texts, train_annotations)

        self.model.config.chunk_long_sequences = True
        self.model.config.max_length = 128

        uncached_preds = self.model.predict(test_texts[:1])

        with self.model.cached_predict():
            start = time.time()
            self.model.predict(test_texts[:1])
            first = time.time()
            self.model.predict(test_texts[:1])
            second = time.time()
            preds = self.model.predict(test_texts[:1])
            assert len(preds) == 1
            preds = self.model.predict(test_texts[:2])
            assert len(preds) == 2

        for uncached_pred, cached_pred in zip(uncached_preds, preds):
            for a, b in zip(
                tf.nest.flatten(uncached_pred), tf.nest.flatten(cached_pred)
            ):
                if a != b and a != approx(b, abs=1e-5):
                    raise Exception("Predictions are not equal")

        first_prediction_time = first - start
        second_prediction_time = second - first
        self.assertLess(second_prediction_time, first_prediction_time / 2.0)

    def test_raises_when_text_doesnt_match(self):
        with self.assertRaises(ValueError):
            self.model.fit(
                ["Text about a dog."],
                [[{"start": 0, "end": 5, "text": "cat", "label": "dog"}]],
            )

    def test_reasonable_predictions(self):
        test_sequence = [
            "I am a dog. A dog that's incredibly bright. I can talk, read, and write!"
        ]
        path = os.path.join(os.path.dirname(__file__), "data", "testdata.json")

        with open(path, "rt") as fp:
            text, labels = json.load(fp)

        self.model.fit(text * 10, labels * 10)

        predictions = self.model.predict(test_sequence)
        self.assertTrue(1 <= len(predictions[0]) <= 3)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in predictions[0]))

        predictions = self.model.predict(test_sequence)
        self.assertTrue(1 <= len(predictions[0]) <= 3)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in predictions[0]))

    def test_chunk_long_sequences(self):
        test_sequence = [
            "I am a dog. A dog that's incredibly bright. I can talk, read, and write! "
            * 10
        ]
        path = os.path.join(os.path.dirname(__file__), "data", "testdata.json")

        # test ValueError raised when raw text is passed along with character idxs and doesn't match
        self.model.config.chunk_long_sequences = True
        self.model.config.max_length = 20

        with open(path, "rt") as fp:
            text, labels = json.load(fp)

        self.model.finetune(text * 10, labels * 10)
        self.assertEqual(self.model.config.dataset_size, 40)

        predictions = self.model.predict(test_sequence)
        self.assertEqual(len(predictions[0]), 20)
        self.assertTrue(any(pred["text"].strip() == "dog" for pred in predictions[0]))

    def test_fit_predict_multi_model(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions
        """
        self.model = SequenceLabeler(
            batch_size=2, max_length=256, lm_loss_coef=0.0, multi_label_sequences=True
        )
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, _ = train_test_split(
            texts, annotations, test_size=0.1
        )
        self.model.fit(train_texts, train_annotations)
        self.model.predict(test_texts)
        probas = self.model.predict_proba(test_texts)
        self.assertIsInstance(probas, list)
        self.assertIsInstance(probas[0], list)
        self.assertIsInstance(probas[0][0], dict)
        self.assertIsInstance(probas[0][0]["confidence"], dict)
        self.model.save(self.save_file)
        model = SequenceLabeler.load(self.save_file)
        model.predict(test_texts)

    def test_pred_alignment(self):
        model = SequenceLabeler(subtoken_predictions=True)
        text = "John J Johnson"
        labels = [{"start": 5, "end": 6, "text": "J", "label": "middle_name"}]
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        self.assertEqual(len(preds), 1)
        del preds[0]["confidence"]
        self.assertEquals(preds, labels)

    def test_bio_tagging(self):
        model = SequenceLabeler(bio_tagging=True)
        text = "Look! Sequential entity-tagging!"
        labels = [
            {"start": 6, "end": 16, "text": "Sequential", "label": "entity"},
            {"start": 17, "end": 24, "text": "entity-", "label": "entity"},
            {"start": 24, "end": 31, "text": "tagging", "label": "entity"},
        ]
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        self.assertEqual(len(preds), 3)
        for p in preds:
            del p["confidence"]
        self.assertEqual(preds, labels)

    def test_auto_negative_chunks(self):
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1, random_state=42
        )
        ans_model = SequenceLabeler(
            max_length=5, chunk_context=0, auto_negative_sampling=True, n_epochs=1
        )
        ans_model.fit(train_texts, train_annotations)
        ans_predictions = ans_model.predict(test_texts)
        ans_token_precision = sequence_labeling_token_precision(
            test_annotations, ans_predictions
        )

        baseline_model = SequenceLabeler(
            max_length=5, chunk_context=0, auto_negative_sampling=False, n_epochs=1
        )
        baseline_model.fit(train_texts, train_annotations)
        baseline_predictions = baseline_model.predict(test_texts)
        baseline_token_precision = sequence_labeling_token_precision(
            test_annotations, baseline_predictions
        )

        assert (
            ans_token_precision["Named Entity"]
            > baseline_token_precision["Named Entity"]
        )

    def test_auto_negative_chunks_chunk_context_default(self):
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1, random_state=42
        )
        ans_model = SequenceLabeler(
            max_length=32, auto_negative_sampling=True, n_epochs=1
        )
        ans_model.fit(train_texts, train_annotations)
        ans_predictions = ans_model.predict(test_texts)

    def test_pre_chunking(self):
        """
        If config.max_document_chars is set, documents are "pre chunked" into "sub
        documents" prior to creation of the input pipeline. This test asserts that
        the splitting and rejoining behaves as expected.
        """
        max_doc_len = 250

        # Create a mix of short and long sequences for prediction
        test_sequence = (
            "I am a dog. A dog that's incredibly bright. I can talk, read, and write! "
        )
        test_sequences = [
            test_sequence,
            test_sequence * 10,
            test_sequence,
            test_sequence * 10,
        ]

        # Use animal test data to train model
        path = os.path.join(os.path.dirname(__file__), "data", "testdata.json")
        with open(path, "rt") as fp:
            text, labels = json.load(fp)
        self.model.finetune(text * 10, labels * 10)

        # Split inputs into "sub documents"
        self.model.config.max_document_chars = max_doc_len
        split_sequences, split_indices = self.model._pre_chunk_document(test_sequences)

        # Predict on the newly split sequences. We unset config.max_document chars
        # prior to calling predict so that we don't repeat the pre chunking that we
        # did explicitly above. This has the potential to be brittle if the functionality
        # of predict in sequence_labeler.py changes
        self.model.config.max_document_chars = None
        split_preds = self.model.predict(split_sequences)

        # Rejoin predictions using split_indices
        self.model.config.max_document_chars = max_doc_len
        preds = self.model._merge_chunked_preds(deepcopy(split_preds), split_indices)

        # Assert that documents were split properly
        assert split_indices == [[0], [1, 2, 3], [4], [5, 6, 7]]
        assert len(split_preds) == 8
        assert len(preds) == 4

        # Assert that the offset is calculated correctly for start/end indices after
        # merging predictions by specifically looking at the last "sub document" of
        # second (index 1) document, which was split into three "sub documents"
        sub_split_preds = split_preds[3]
        sub_preds = [pred for pred in preds[1] if pred["start"] >= max_doc_len * 2]
        for split_pred, pred in zip(sub_split_preds, sub_preds):
            assert split_pred["start"] + max_doc_len * 2 == pred["start"]
            assert split_pred["end"] + max_doc_len * 2 == pred["end"]

    def test_pre_chunking_neg_confidences(self):
        """
        Similar to test_pre_chunking(), except with running predict w/ the
        return_negative_confidence flag set to True. This changes the return format
        of prediction from List[List[Dict]] to List[Dict], so hits a different
        code path in merging of chunks that needs to be tested.
        """
        max_doc_len = 250

        # Create a mix of short and long sequences for prediction
        test_sequence = (
            "I am a dog. A dog that's incredibly bright. I can talk, read, and write! "
        )
        test_sequences = [test_sequence * 10]

        # Use animal test data to train model
        path = os.path.join(os.path.dirname(__file__), "data", "testdata.json")
        with open(path, "rt") as fp:
            text, labels = json.load(fp)
        self.model.finetune(text * 10, labels * 10)

        # Predict w/ max_document_chars set
        self.model.config.max_document_chars = max_doc_len
        preds_mdc = self.model.predict(test_sequences, return_negative_confidence=True)

        # Predict w/o max_document_chars set
        self.model.config.max_document_chars = None
        preds = self.model.predict(test_sequences, return_negative_confidence=True)

        # Verify that the indices line up
        for pred_mdc, pred in zip(preds_mdc[0]["prediction"], preds[0]["prediction"]):
            assert pred_mdc["start"] == pred["start"]
            assert pred_mdc["end"] == pred["end"]

    def test_max_document_chars(self):
        """
        If documents are "pre chunked" due to config.max_document_chars being set,
        predictions are impacted when a document is split in the middle of a non-pad
        token. This test asserts that the impact is minimal.
        """
        # Use animal test data to train model
        test_sequence = [
            "I am a dog. A dog that's incredibly bright. I can talk, read, and write! "
            * 10
        ]
        path = os.path.join(os.path.dirname(__file__), "data", "testdata.json")
        with open(path, "rt") as fp:
            text, labels = json.load(fp)
        self.model.finetune(text * 10, labels * 10)

        # Predict without config.max_document_chars
        preds = self.model.predict(test_sequence)

        # Predict with config.max_document_chars set to split long sequence
        self.model.config.max_document_chars = 250
        pre_chunk_preds = self.model.predict(test_sequence)

        # Ensure that the number of predictions of the span "dog" is almost identical
        # between preds and pre_chunk_preds. The reason these values are not identical
        # is because with pre chunking, "dog" can be split across multiple sub documents
        num_dog_preds = len([p for p in preds[0] if p["text"] == "dog"])
        num_dog_pre_chunk_preds = len(
            [p for p in pre_chunk_preds[0] if p["text"] == "dog"]
        )
        assert num_dog_pre_chunk_preds / num_dog_preds >= 0.95

    def test_labeled_whitespace(self):
        model = SequenceLabeler()
        text = (
            " I am a dog. A dog that's incredibly bright. I can talk, read, and write!"
        )
        labels = [{"start": 0, "end": 8, "text": " I am a ", "label": "entity"}]
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        self.assertEqual(len(preds), 1)
        for p in preds:
            del p["confidence"]
        correct_label = [{"start": 1, "end": 7, "text": "I am a", "label": "entity"}]
        self.assertEqual(preds, correct_label)


class TestSequenceMemoryLeak(unittest.TestCase):
    @staticmethod
    def is_wr(val):
        return val is not None and not isinstance(val, (int, float, str))

    @staticmethod
    def get_weakrefs(dictionary):
        weakrefs = []
        for v in dictionary.values():
            if hasattr(v, "__dict__"):
                weakrefs += TestSequenceMemoryLeak.get_weakrefs(v.__dict__)
            if isinstance(v, dict):
                weakrefs += TestSequenceMemoryLeak.get_weakrefs(v)
            elif isinstance(v, (list, tuple)):
                for vi in v:
                    if isinstance(vi, dict):
                        weakrefs += TestSequenceMemoryLeak.get_weakrefs(vi)
                    elif TestSequenceMemoryLeak.is_wr(vi):
                        try:
                            weakrefs.append(weakref.ref(vi))
                        except Exception as e:
                            print(e)
            elif TestSequenceMemoryLeak.is_wr(v):
                try:
                    weakrefs.append(weakref.ref(v))
                except Exception as e:
                    print(e)
        return weakrefs

    def test_leaking_objects(self):
        previous_model_wrs = None
        for _ in range(10):
            model = SequenceLabeler(n_epochs=1)
            model.fit(["some text"], [[]])
            wrs = self.get_weakrefs(model.__dict__)
            del model
            tf.compat.v1.reset_default_graph()
            gc.collect()
            if previous_model_wrs is None:
                previous_model_wrs = [w for w in wrs if w() is not None]
            else:
                new_refs = [w() for w in wrs if w() is not None]
                prevous_refs = [w() for w in previous_model_wrs]
                for new in new_refs:
                    # Assert that no new objects are introduced that cannot be cleaned up.
                    print(new)
                    self.assertTrue(any(new is old for old in prevous_refs))

    def test_pred_alignment(self):
        model = SequenceLabeler(subtoken_predictions=True)
        text = "John J Johnson"
        labels = [{"start": 5, "end": 6, "text": "J", "label": "middle_name"}]
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        self.assertEqual(len(preds), 1)
        del preds[0]["confidence"]
        self.assertEquals(preds, labels)

    def test_auto_negative_chunks(self):
        raw_docs = ["".join(text) for text in self.texts]
        texts, annotations = finetune_to_indico_sequence(
            raw_docs, self.texts, self.labels, none_value=self.model.config.pad_token
        )
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=0.1, random_state=42
        )

        ans_model = SequenceLabeler(
            max_length=5, chunk_context=0, auto_negative_sampling=True, n_epochs=1
        )
        ans_model.fit(train_texts, train_annotations)
        ans_predictions = ans_model.predict(test_texts)
        ans_token_precision = sequence_labeling_token_precision(
            test_annotations, ans_predictions
        )

        baseline_model = SequenceLabeler(
            max_length=5, chunk_context=0, auto_negative_sampling=False, n_epochs=1
        )
        baseline_model.fit(train_texts, train_annotations)
        baseline_predictions = baseline_model.predict(test_texts)
        baseline_token_precision = sequence_labeling_token_precision(
            test_annotations, baseline_predictions
        )

        assert (
            ans_token_precision["Named Entity"]
            > baseline_token_precision["Named Entity"]
        )


class TestSequenceMemoryLeak(unittest.TestCase):
    @staticmethod
    def is_wr(val):
        return val is not None and not isinstance(val, (int, float, str))

    @staticmethod
    def get_weakrefs(dictionary):
        weakrefs = []
        for v in dictionary.values():
            if hasattr(v, "__dict__"):
                weakrefs += TestSequenceMemoryLeak.get_weakrefs(v.__dict__)
            if isinstance(v, dict):
                weakrefs += TestSequenceMemoryLeak.get_weakrefs(v)
            elif isinstance(v, (list, tuple)):
                for vi in v:
                    if isinstance(vi, dict):
                        weakrefs += TestSequenceMemoryLeak.get_weakrefs(vi)
                    elif TestSequenceMemoryLeak.is_wr(vi):
                        try:
                            weakrefs.append(weakref.ref(vi))
                        except Exception as e:
                            print(e)
            elif TestSequenceMemoryLeak.is_wr(v):
                try:
                    weakrefs.append(weakref.ref(v))
                except Exception as e:
                    print(e)
        return weakrefs

    def test_leaking_objects(self):
        previous_model_wrs = None
        for _ in range(10):
            model = SequenceLabeler(n_epochs=1)
            model.fit(["some text"], [[]])
            wrs = self.get_weakrefs(model.__dict__)
            del model
            tf.compat.v1.reset_default_graph()
            gc.collect()
            if previous_model_wrs is None:
                previous_model_wrs = [w for w in wrs if w() is not None]
            else:
                new_refs = [w() for w in wrs if w() is not None]
                prevous_refs = [w() for w in previous_model_wrs]
                for new in new_refs:
                    # Assert that no new objects are introduced that cannot be cleaned up.
                    print(new)
                    self.assertTrue(any(new is old for old in prevous_refs))


class TestSequenceLabelerNoCRF(TestSequenceLabeler):
    def default_config(self, **kwargs):
        d = dict(
            base_model=GPT,
            batch_size=2,
            max_length=256,
            lm_loss_coef=0.0,
            val_size=0,
            crf_sequence_labeling=False,
        )
        d.update(**kwargs)
        return d

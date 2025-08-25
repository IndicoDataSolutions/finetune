import codecs
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import requests
import tensorflow as tf
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from pytest import approx
from sequence_metrics.metrics import (
    sequence_labeling_overlap_precision,
    sequence_labeling_overlap_recall,
    sequence_labeling_token_precision,
    sequence_labeling_token_recall,
)
from sklearn.model_selection import train_test_split

import pytest

from finetune import SequenceLabeler
from finetune.base_models import GPT, TestingModel
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence

def test_fit_predict_real(get_untrained_sequence_labeler, reuters_indico_sequence, tmp_path):
    train_texts, test_texts, train_annotations, test_annotations = train_test_split(
        *reuters_indico_sequence, test_size=0.1, random_state=42
    )
    reweighted_model = get_untrained_sequence_labeler(class_weights={"Named Entity": 10.0})
    sequence_model = get_untrained_sequence_labeler()
    reweighted_model.fit(train_texts, train_annotations)
    reweighted_predictions = reweighted_model.predict(test_texts)
    reweighted_token_recall = sequence_labeling_token_recall(
        test_annotations, reweighted_predictions
    )

    sequence_model.fit(train_texts, train_annotations)
    predictions = sequence_model.predict(test_texts)
    _ = sequence_model.predict(test_texts, per_token=True)
    with_doc_probas = sequence_model.predict(
        test_texts, return_negative_confidence=True
    )
    probas = sequence_model.predict_proba(test_texts)

    for pred, pred_with_prob in zip(predictions, with_doc_probas):
        assert pred == pred_with_prob["prediction"]
        assert isinstance(pred_with_prob["negative_confidence"], dict)

    assert isinstance(probas, list)
    assert isinstance(probas[0], list)
    assert isinstance(probas[0][0], dict)
    assert isinstance(probas[0][0]["confidence"], dict)

    token_precision = sequence_labeling_token_precision(test_annotations, predictions)
    token_recall = sequence_labeling_token_recall(test_annotations, predictions)
    overlap_precision = sequence_labeling_overlap_precision(
        test_annotations, predictions
    )
    overlap_recall = sequence_labeling_overlap_recall(test_annotations, predictions)

    assert "Named Entity" in token_precision
    assert "Named Entity" in token_recall
    assert "Named Entity" in overlap_precision
    assert "Named Entity" in overlap_recall
    assert reweighted_token_recall["Named Entity"] > token_recall["Named Entity"]


def test_raises_when_text_doesnt_match(get_untrained_sequence_labeler):
    model = get_untrained_sequence_labeler()
    with pytest.raises(ValueError):
        model.fit(
            ["Text about a dog."],
            [[{"start": 0, "end": 5, "text": "cat", "label": "dog"}]],
        )


def test_auto_negative_chunks(mock_get_keras_model, get_untrained_sequence_labeler, reuters_indico_sequence, ):    
    model = get_untrained_sequence_labeler(auto_negative_sampling=True)
    assert model.config.auto_negative_sampling
    assert model.config.chunk_long_sequences
    model.fit(reuters_indico_sequence[0][:5], reuters_indico_sequence[1][:5])
    assert len(mock_get_keras_model) == 2 # One for the sampling and one for the final model
    sampling_model = mock_get_keras_model[0]
    assert sampling_model.config.max_empty_chunk_ratio == 0.0
    assert not sampling_model.config.auto_negative_sampling
    assert sampling_model.config is not model.config

    # Check that the sampling model was called with the correct data
    assert len(sampling_model.fit_calls) == 1
    assert len(sampling_model.predict_calls) > 1

    final_model = mock_get_keras_model[1]
    assert final_model.config is model.config
    assert len(final_model.fit_calls) == 1
    assert len(final_model.predict_calls) == 0



def test_pre_chunking(get_untrained_sequence_labeler):
    max_doc_len = 250
    test_sequence = (
        "I am a dog. A dog that's incredibly bright. I can talk, read, and write! "
    )
    test_sequences = [test_sequence, test_sequence * 10, test_sequence, test_sequence * 10]
    sequence_model = get_untrained_sequence_labeler()
    sequence_model.config.max_document_chars = max_doc_len

    assert any(len(seq) > max_doc_len for seq in test_sequences) # Just to check that we are using data that is long enough to be chunked
    split_sequences, split_indices = sequence_model._pre_chunk_document(test_sequences)
    assert all(len(seq) <= max_doc_len for seq in split_sequences)

    assert len(split_indices) ==  len(test_sequences)
    for split_idxs, input_seq in zip(split_indices, test_sequences):
        if len(input_seq) < max_doc_len:
            assert len(split_idxs) == 1
            assert input_seq == split_sequences[split_idxs[0]]
        else:
            assert len(split_idxs) > 1
            assert "".join(split_sequences[split_idx] for split_idx in split_idxs) == input_seq
        assert max(split_idxs) < len(split_sequences)

    mock_preds = [
        [{"start": 0, "end": 5, "text": seq[0: 5], "label": i}] for i, seq in enumerate(split_sequences)
    ]
    assert len(mock_preds) == len(split_sequences)

    merged_preds = sequence_model._merge_chunked_preds(mock_preds, split_indices)
    assert len(merged_preds) == len(test_sequences)
    for pred, input_seq, split_idxs in zip(merged_preds, test_sequences, split_indices):
        assert len(pred) == len(split_idxs)
        assert set(l["label"] for l in pred) == set(split_idxs)
        for l, split_idx in zip(pred, split_idxs):
            # check the offsets have been correctly adjusted
            assert l["text"] == input_seq[l["start"]: l["end"]]
            assert l["text"] == mock_preds[split_idx][0]["text"] # check that the text is the same across the merging operation
        
        assert all(l["end"] - l["start"] == 5 for l in pred) # check that the preds are still all the right length
        if len(split_idxs) > 1:
            # Check this is not a trivial solution and that some of the offsets have needed to be adjusted
            assert len([l for l in pred if l["start"] != 0]) == len(split_idxs) - 1
        else:
            assert pred[0]["start"] == 0


def test_pre_chunking_neg_confidences(get_untrained_sequence_labeler):
    max_doc_len = 250
    test_sequence = (
        "I am a dog. A dog that's incredibly bright. I can talk, read, and write! "
    )
    test_sequences = [test_sequence, test_sequence * 10, test_sequence, test_sequence * 10]
    sequence_model = get_untrained_sequence_labeler()
    sequence_model.config.max_document_chars = max_doc_len
    split_sequences, split_indices = sequence_model._pre_chunk_document(test_sequences)
    mock_preds = [
        {"prediction": [{"start": 0, "end": 5, "text": seq[0: 5], "label": "Label"}], "negative_confidence": {"Label": 1 / (i + 1)}} for i, seq in enumerate(split_sequences)
    ]
    assert len(mock_preds) == len(split_sequences)  

    merged_preds = sequence_model._merge_chunked_preds(mock_preds, split_indices, return_negative_confidence=True)
    assert len(merged_preds) == len(test_sequences)

    for pred, input_seq, split_idxs in zip(merged_preds, test_sequences, split_indices):
        predictions = pred["prediction"]
        negative_confidence = pred["negative_confidence"]
        assert len(predictions) == len(split_idxs)
        assert negative_confidence["Label"] == 1 / (min(split_idxs) + 1) # Check that we have max reduced the negative confidence

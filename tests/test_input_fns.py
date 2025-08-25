import numpy as np
import tensorflow as tf
import pytest

from finetune import Classifier, SequenceLabeler
from finetune.util.input_utils import InputMode, Chunker
from finetune.encoding.input_encoder import EncodedOutput
from finetune.errors import FinetuneError


def _make_encoded_output(seq_len=8, include_special=False, offset=0):
    tokens = ["a"] * seq_len
    token_ids = np.arange(seq_len)
    token_starts = np.array([max(i - 1, 0) + offset for i in range(seq_len)], dtype=int)
    token_ends = np.array([i + offset for i in range(seq_len)], dtype=int)
    if include_special:
        token_starts[0] = -1
        token_ends[-1] = -1
    return EncodedOutput(
        token_ids=np.asarray(token_ids),
        tokens=np.asarray(tokens),
        token_ends=np.asarray(token_ends),
        token_starts=np.asarray(token_starts),
        useful_start=0,
        useful_end=seq_len,
    )


@pytest.fixture
def patch__text_to_ids(monkeypatch):
    calls = {"count": 0}

    def _apply(pipeline, chunks_per_doc=1, seq_len=8):
        def fake_ids(Xs, pad_token=None):
            for _ in range(chunks_per_doc):
                calls["count"] += 1
                yield _make_encoded_output(seq_len=seq_len)

        monkeypatch.setattr(pipeline, "_text_to_ids", fake_ids)
        return calls

    return _apply


@pytest.fixture
def patch_batch_dataset(monkeypatch):
    recorded = {"kwargs": None}

    def fake_batch_dataset(dataset, *args, **kwargs):
        recorded["kwargs"] = kwargs
        return dataset

    monkeypatch.setattr("finetune.input_pipeline.batch_dataset", fake_batch_dataset)
    return recorded


def test_input_spec_shapes_classification(get_untrained_classifier):
    model = get_untrained_classifier(max_length=16, batch_size=4)
    pipeline = model.input_pipeline

    types, shapes = pipeline.input_spec(concrete_dims=False, include_targets=False, batched=False)
    assert "tokens" in types and "tokens" in shapes
    assert shapes["tokens"].rank == 1 and shapes["tokens"].as_list() == [None]

    (types_b, target_type), (shapes_b, target_shape) = pipeline.input_spec(
        concrete_dims=True, include_targets=True, batched=True
    )
    assert shapes_b["tokens"].as_list() == [model.config.batch_size, model.config.max_length]
    assert target_shape.rank == 2 and target_shape.as_list()[0] == model.config.batch_size


def test_input_spec_shapes_sequence(get_untrained_sequence_labeler):
    model = get_untrained_sequence_labeler(max_length=32, batch_size=2)
    pipeline = model.input_pipeline

    (types_b, target_type), (shapes_b, target_shape) = pipeline.input_spec(
        concrete_dims=True, include_targets=True, batched=True
    )
    assert target_shape.as_list() == [model.config.batch_size, model.config.max_length]


def test_zip_list_to_dict_errors(get_untrained_classifier):
    model = get_untrained_classifier()
    pipeline = model.input_pipeline
    with pytest.raises(FinetuneError):
        pipeline.zip_list_to_dict(["a", "b"], Y=[1])
    with pytest.raises(FinetuneError):
        pipeline.zip_list_to_dict(["a"], Y=None, context=[[0], [1]])


def test_get_dataset_from_list_batches_and_size(get_untrained_classifier, patch__text_to_ids, patch_batch_dataset):
    model = get_untrained_classifier(max_length=16, batch_size=3, n_epochs=2)
    pipeline = model.input_pipeline

    X = ["doc1", "doc2", "doc3", "doc4"]
    Y = ["A", "B", "A", "B"]
    data_list = pipeline.zip_list_to_dict(X, Y)

    patch__text_to_ids(pipeline, chunks_per_doc=2, seq_len=6)

    out = pipeline.get_dataset_from_list(data_list, InputMode.TRAIN, update_hook=None)
    assert "train_dataset" in out
    assert model.config.dataset_size == 8

    recorded = patch_batch_dataset
    assert recorded["kwargs"]["batch_size"] == model.config.batch_size
    assert recorded["kwargs"]["max_length"] == model.config.max_length
    assert recorded["kwargs"]["n_epochs"] == model.config.n_epochs
    assert recorded["kwargs"]["drop_remainder"] is True


def test_get_dataset_from_generator_predict(get_untrained_classifier, patch__text_to_ids, patch_batch_dataset):
    model = get_untrained_classifier(max_length=12, predict_batch_size=5)
    pipeline = model.input_pipeline
    patch__text_to_ids(pipeline, chunks_per_doc=1, seq_len=5)

    def gen():
        for x in ["a", "b", "c"]:
            yield {"X": x}

    out = pipeline.get_dataset_from_generator(gen, InputMode.PREDICT, update_hook=None)
    assert "predict_dataset" in out
    recorded = patch_batch_dataset
    assert recorded["kwargs"]["batch_size"] == model.config.predict_batch_size
    assert recorded["kwargs"]["table_batching"] is False


def test_chunker_basic():
    c = Chunker(max_length=10, total_context_width=4, justify="center")
    assert c.useful_chunk_width == 4
    chunks = list(c.generate_chunks(length=13))
    assert len(chunks) == 3
    (s0, e0, (u0s, u0e)) = chunks[0]
    # For start_of_doc, useful_start == 0, useful_end == c.normal_end
    assert u0s == 0 and u0e == c.normal_end
    (s1, e1, (u1s, u1e)) = chunks[1]
    assert (u1s, u1e) == (c.normal_start, c.normal_end)
    (s2, e2, (u2s, u2e)) = chunks[-1]
    # For end_of_doc, useful_end == max_length
    assert u2e == c.max_length
 

def test_input_spec_with_aux_context_shapes(get_untrained_classifier):
    model = get_untrained_classifier(max_length=20, batch_size=3)
    model.config.use_auxiliary_info = True
    model.config.context_dim = 5
    pipeline = model.input_pipeline

    (types_b, target_type), (shapes_b, target_shape) = pipeline.input_spec(
        concrete_dims=True, include_targets=True, batched=True
    )
    assert "context" in shapes_b
    assert shapes_b["context"].as_list() == [model.config.batch_size, model.config.max_length, model.config.context_dim]


def test_train_dataset_runnable_and_shapes_with_context(get_untrained_classifier, patch__text_to_ids, monkeypatch):
    model = get_untrained_classifier(max_length=16, batch_size=2, n_epochs=1)
    model.config.use_auxiliary_info = True
    model.config.context_dim = 4
    pipeline = model.input_pipeline

    # Patch tokenization to generate context of the right shape regardless of input
    def fake_tokenize_context(context, encoded_output, config):
        seq_len = len(encoded_output.token_ids)
        return np.zeros((seq_len, config.context_dim), dtype=np.float32)

    monkeypatch.setattr("finetune.input_pipeline.tokenize_context", fake_tokenize_context)

    # Generate 3 docs so we get 1 full batch and 1 dropped sample due to drop_remainder=True
    X = ["d1", "d2", "d3"]
    Y = ["A", "B", "A"]
    C = [[{"end": 1}], [{"end": 1}], [{"end": 1}]]
    data_list = pipeline.zip_list_to_dict(X, Y, context=C)

    # One chunk per doc
    patch__text_to_ids(pipeline, chunks_per_doc=1, seq_len=7)

    out = pipeline.get_dataset_from_list(data_list, InputMode.TRAIN)
    ds = out["train_dataset"]
    # Iterate one batch and check shapes are static [batch_size, max_length]
    for (feats, targets) in ds.take(1):
        assert set(feats.keys()) >= {"tokens", "length", "context"}
        assert feats["tokens"].shape.as_list() == [model.config.batch_size, model.config.max_length]
        assert feats["context"].shape.as_list() == [model.config.batch_size, model.config.max_length, model.config.context_dim]
        assert feats["length"].shape.as_list() == [model.config.batch_size]
        # targets should be [batch_size, target_dim]
        assert targets.shape.rank == 2 and targets.shape.as_list()[0] == model.config.batch_size


def test_predict_dataset_runnable_and_limits(get_untrained_classifier, patch__text_to_ids):
    model = get_untrained_classifier(max_length=18, predict_batch_size=4)
    pipeline = model.input_pipeline
    patch__text_to_ids(pipeline, chunks_per_doc=1, seq_len=9)

    def gen():
        for x in ["a", "bb", "ccc", "dddd", "eeeee"]:
            yield {"X": x}

    out = pipeline.get_dataset_from_generator(gen, InputMode.PREDICT)
    pds = out["predict_dataset"]
    # Take a batch and verify dynamic dims are <= configured sizes
    for feats in pds.take(1):
        tokens = feats["tokens"]
        length = feats["length"]
        # Batch dim <= predict_batch_size
        assert tf.shape(tokens)[0] <= model.config.predict_batch_size
        # Sequence dim <= max_length
        assert tf.shape(tokens)[1] <= model.config.max_length
        # length vector also respects batch dimension
        assert tf.shape(length)[0] == tf.shape(tokens)[0]


def test_class_weights_computation_linear(get_untrained_classifier, patch__text_to_ids):
    model = get_untrained_classifier(max_length=16, batch_size=2, n_epochs=1)
    # Request computation of class weights
    model.config.class_weights = "linear"
    pipeline = model.input_pipeline

    X = ["doc1", "doc2", "doc3"]
    Y = ["A", "A", "B"]
    data_list = pipeline.zip_list_to_dict(X, Y)

    # Ensure one chunk per doc
    patch__text_to_ids(pipeline, chunks_per_doc=1, seq_len=6)

    pipeline.get_dataset_from_list(data_list, InputMode.TRAIN)
    # Expect class weights dict with ratios max_count/count -> A: 1.0, B: 2.0
    cw = model.config.class_weights
    assert isinstance(cw, dict)
    assert set(cw.keys()) >= {"A", "B"}
    assert pytest.approx(cw["A"], rel=1e-6) == 1.0
    assert pytest.approx(cw["B"], rel=1e-6) == 2.0

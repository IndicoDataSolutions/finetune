from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

import finetune.scheduler as scheduler_mod
from finetune.base_models import SourceModel
from finetune.base_models.bert.table_utils import TableModelBatchPostprocessor
from finetune.errors import FinetuneSchedulerError
from finetune.scheduler import Scheduler


class DummySourceModelA(SourceModel):
    encoder = None
    featurizer = None
    settings = {}


class DummySourceModelB(SourceModel):
    encoder = None
    featurizer = None
    settings = {}


class DummyLoadedModel:
    def __init__(self, base_model, outcomes=None):
        self.config = SimpleNamespace(base_model=base_model)
        self.saver = SimpleNamespace(variables={}, fallback_=object())
        self._outcomes = list(outcomes or [["ok"]])
        self.close_calls = 0
        self._cached_predict = False

    def predict(self, x, *args, **kwargs):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self, update_saver=False):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def patch_cleanup(monkeypatch):
    monkeypatch.setattr(scheduler_mod.MODEL_REGISTRY, "cleanup", lambda: None)


@pytest.fixture
def fake_memory(monkeypatch):
    state = {"in_use": 0, "peak": 0, "gpu_available": True, "cpu_percent": 10}

    monkeypatch.setattr(
        scheduler_mod,
        "is_gpu_available",
        lambda: state["gpu_available"],
    )
    monkeypatch.setattr(scheduler_mod, "BytesInUse", lambda: state["in_use"])
    monkeypatch.setattr(scheduler_mod, "MaxBytesInUse", lambda: state["peak"])
    monkeypatch.setattr(scheduler_mod, "BytesLimit", lambda: 1000)
    monkeypatch.setattr(
        scheduler_mod.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=state["cpu_percent"]),
    )
    return state


def test_memory_windows_are_bounded_per_source_model():
    shed = Scheduler(memory_window_size=3)

    for sample in (100, 10, 9, 8):
        shed._record_prediction_memory_sample(
            "DummySourceModelA", max_above_resting=sample, model_size=sample // 2
        )
    shed._record_prediction_memory_sample(
        "DummySourceModelB", max_above_resting=5, model_size=7
    )

    assert shed._estimated_memory_stats("DummySourceModelA") == {
        "max_above_resting": 10,
        "model_size": 5,
    }
    assert shed._estimated_memory_stats("DummySourceModelB") == {
        "max_above_resting": 5,
        "model_size": 7,
    }
    assert shed._estimated_memory_stats("unknown") == {
        "max_above_resting": 9,
        "model_size": 7,
    }


def test_cache_miss_loads_first_and_uses_source_specific_stats(
    monkeypatch, fake_memory
):
    shed = Scheduler(reserved=0, execution_safety_margin=0)
    shed.gpu_memory_limit = 1000
    shed._record_prediction_memory_sample(
        "UnrelatedSource", max_above_resting=900, model_size=50
    )

    keep = DummyLoadedModel(DummySourceModelB)
    shed.model_cache["keep"] = keep
    shed.loaded_models.append("keep")
    fake_memory["in_use"] = 100

    loaded = []

    def fake_load(model_file, key=None, **kwargs):
        model = DummyLoadedModel(DummySourceModelA)
        loaded.append((model_file, key, model))
        return model

    monkeypatch.setattr(scheduler_mod.BaseModel, "load", fake_load)

    out_model = shed.get_model("fresh.jl")

    assert out_model is loaded[0][2]
    assert shed.cache_key_to_source_model["model=fresh.jl"] == "DummySourceModelA"
    assert shed.loaded_models == ["keep", "model=fresh.jl"]
    assert keep.close_calls == 0


def test_cache_hit_execution_headroom_evicts_unrelated_models(monkeypatch, fake_memory):
    shed = Scheduler(reserved=0, execution_safety_margin=0)
    shed.gpu_memory_limit = 1000
    shed._record_prediction_memory_sample(
        "DummySourceModelA", max_above_resting=400, model_size=0
    )

    other = DummyLoadedModel(DummySourceModelB)
    active = DummyLoadedModel(DummySourceModelA)
    shed.model_cache = {"other": other, "active": active}
    shed.loaded_models = ["other", "active"]
    shed.cache_key_to_source_model = {
        "other": "DummySourceModelB",
        "active": "DummySourceModelA",
    }

    fake_memory["in_use"] = 700

    def dynamic_in_use():
        return 700 if "other" in shed.loaded_models else 200

    monkeypatch.setattr(scheduler_mod, "BytesInUse", dynamic_in_use)

    shed._ensure_prediction_headroom(
        source_model_key="DummySourceModelA",
        active_cache_key="active",
        include_model_size=False,
        allow_global_fallback=False,
    )

    assert shed.loaded_models == ["active"]
    assert other.close_calls == 1


def test_large_transient_models_run_exclusively(fake_memory):
    shed = Scheduler(reserved=0, execution_safety_margin=0)
    shed.gpu_memory_limit = 1000
    shed._record_prediction_memory_sample(
        "DummySourceModelA", max_above_resting=600, model_size=0
    )

    older = DummyLoadedModel(DummySourceModelB)
    newer = DummyLoadedModel(DummySourceModelB)
    active = DummyLoadedModel(DummySourceModelA)
    shed.model_cache = {"older": older, "newer": newer, "active": active}
    shed.loaded_models = ["older", "newer", "active"]
    shed.cache_key_to_source_model = {
        "older": "DummySourceModelB",
        "newer": "DummySourceModelB",
        "active": "DummySourceModelA",
    }
    fake_memory["in_use"] = 100

    shed._ensure_prediction_headroom(
        source_model_key="DummySourceModelA",
        active_cache_key="active",
        include_model_size=False,
        allow_global_fallback=False,
    )

    assert shed.loaded_models == ["active"]
    assert older.close_calls == 1
    assert newer.close_calls == 1


def test_scheduler_retries_after_exception(monkeypatch, fake_memory):
    fake_memory["gpu_available"] = False
    shed = Scheduler()
    loads = []

    def fake_load(model_file, key=None, **kwargs):
        outcomes = [RuntimeError("oom-ish failure")] if not loads else [["ok"]]
        model = DummyLoadedModel(DummySourceModelA, outcomes)
        loads.append(model)
        return model

    monkeypatch.setattr(scheduler_mod.BaseModel, "load", fake_load)

    result = shed.predict("dummy.jl", ["A"])

    assert result == ["ok"]
    assert len(loads) == 2
    assert loads[0].close_calls == 1
    assert shed.loaded_models == ["model=dummy.jl"]


def test_scheduler_wraps_retry_failure(monkeypatch, fake_memory):
    fake_memory["gpu_available"] = False
    shed = Scheduler()

    def fake_load(model_file, key=None, **kwargs):
        return DummyLoadedModel(
            DummySourceModelA,
            [RuntimeError("first fail"), RuntimeError("second fail")],
        )

    monkeypatch.setattr(scheduler_mod.BaseModel, "load", fake_load)

    with pytest.raises(FinetuneSchedulerError) as excinfo:
        shed.predict("dummy.jl", ["A"])

    message = str(excinfo.value)
    assert "Original Error: first fail" in message
    assert "Retry Error:" in message


def test_update_memory_limit_uses_per_process_fraction(fake_memory):
    shed = Scheduler(config={"per_process_gpu_memory_fraction": 0.4})
    model = DummyLoadedModel(DummySourceModelA)

    shed._update_memory_limit(model)

    assert shed.gpu_memory_limit == 400


def test_table_predict_batch_postprocessor_matches_direct_postprocess():
    config = SimpleNamespace(
        context_dim=4,
        chunk_tables=True,
        predict_batch_size=1,
        max_length=2048,
    )
    postprocessor = TableModelBatchPostprocessor(config=config)

    features = {
        "tokens": np.array([1, 2, 3, 4], dtype=np.int32),
        "context": np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 2, 0],
                [1, 0, 3, 0],
            ],
            dtype=np.float32,
        ),
    }

    batch = next(
        postprocessor.iter_predict_batches(iter([features]), predict_batch_size=1)
    )
    direct = postprocessor._postprocess(
        {
            "tokens": tf.convert_to_tensor([[1, 2, 3, 4]], dtype=tf.int32),
            "context": tf.convert_to_tensor([features["context"]], dtype=tf.float32),
            "length": tf.convert_to_tensor([4], dtype=tf.int32),
        },
        training=False,
    )

    for key in ("tokens", "context", "length"):
        np.testing.assert_array_equal(batch[key].numpy(), direct[key].numpy())
    for key in ("row_gather", "col_gather"):
        for inner_key in ("seq_lens", "values", "attn_mask", "pos_ids"):
            np.testing.assert_array_equal(
                batch[key][inner_key].numpy(), direct[key][inner_key].numpy()
            )

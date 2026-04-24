import functools
import gc
import logging
import sys
from collections import OrderedDict, defaultdict, deque

import psutil
import pynvml
import tensorflow as tf

from finetune.base import MODEL_REGISTRY, BaseModel
from finetune.errors import FinetuneSchedulerError
from finetune.target_models.sequence_labeling import SequenceLabeler

LOGGER = logging.getLogger("finetune")


def BytesInUse():
    """Generates an op that computes the current memory of a device."""
    return tf.config.experimental.get_memory_info("GPU:0")["current"]


def BytesLimit():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.total


def MaxBytesInUse():
    """Generates an op that computes the peak memory of a device."""
    return tf.config.experimental.get_memory_info("GPU:0")["peak"]


def is_gpu_available():
    return len(tf.config.list_physical_devices("GPU")) > 0


def bytes_to_meg(x):
    return x / 1024 / 1024


def scheduled(fn):
    @functools.wraps(fn)
    def scheduled_predict(
        self, model_file, x, *args, key=None, config_overrides=None, **kwargs
    ):
        # this is just for backwards compat, should always have a blob key going forward
        cache_key = kwargs.pop("cache_key", None)
        resolved_cache_key = self.model_cache_key(
            model_file, key=key, cache_key=cache_key
        )
        model = self._rotate_in_model(
            model_file, key=key, config_overrides=config_overrides, cache_key=cache_key
        )
        self._reset_peak_memory_stats()
        try:
            preds = fn(self, model_file=model_file, x=x, *args, model=model, **kwargs)
        except Exception as orig_except:
            LOGGER.warning(
                "Exception '{}' raised. Closing all models and retrying".format(
                    orig_except
                )
            )
            # Close everything to make sure we have available memory
            self.close_all()
            try:
                # Reload in preparation for prediction
                model = self._rotate_in_model(
                    model_file,
                    key=key,
                    config_overrides=config_overrides,
                    cache_key=cache_key,
                )
                self._reset_peak_memory_stats()
                preds = fn(
                    self, model_file=model_file, x=x, *args, model=model, **kwargs
                )
            except Exception as e:
                raise FinetuneSchedulerError(
                    "Original Error: {}, Retry Error: {}".format(
                        str(orig_except), str(e)
                    )
                )
        self._record_prediction_memory(model, resolved_cache_key)
        self._update_memory_limit(model)
        return preds

    return scheduled_predict


class Scheduler:
    # Scheduler behavior overview:
    #
    # 1. Models are cached by resolved cache key (`model_cache_key`) and tracked in
    #    `loaded_models` as an LRU queue. Cache hits move the active key to the end
    #    of that queue. Cache misses load the model, then register its base model
    #    type so future memory estimates are source-specific.
    #
    # 2. GPU admission is based on recent observed memory for the model's
    #    `config.base_model` type, not a single global historical max. For each
    #    source model type we keep a bounded window of recent:
    #       - `max_above_resting`: transient peak above steady-state memory during
    #         prediction
    #       - `model_size`: approximate resident memory added by loading the model
    #    This gives fast recovery from one-off spikes while still preserving a
    #    conservative running max for recent behavior.
    #
    # 3. On cache miss, the model is loaded before the final GPU headroom decision.
    #    That lets us discover the source model type and use its own memory history
    #    instead of falling back to unrelated global spikes. CPU pressure is still
    #    checked first so we can evict before loading if host RAM is tight.
    #
    # 4. Before every prediction, including cache hits, we run an execution-time
    #    headroom check. If the active model's expected transient execution memory
    #    does not fit alongside currently loaded peers, we evict older unrelated
    #    models until it does. Very large transient models are forced into
    #    effectively exclusive execution because allocator fragmentation can make a
    #    "fits on paper" run still fail in practice.
    #
    # 5. Peak memory statistics are reset around each prediction attempt, then the
    #    successful attempt records fresh memory samples back into the per-source
    #    windows. If prediction still fails, we keep the existing recovery path:
    #    close all models, reload, and retry once before raising a
    #    `FinetuneSchedulerError`.
    def __init__(
        self,
        max_models=None,
        config=None,
        reserved=750000000,
        ram_max_frac=0.8,
        memory_window_size=10,
        execution_safety_margin=512 * 1024 * 1024,
    ):
        self.loaded_models = list()
        self.max_models = max_models
        self.gpu_memory_limit = None
        self.model_cache = dict()
        self.max_above_resting = None
        self.max_model_size = None
        self.config = config or {}
        self.reserved = reserved
        self.ram_max_frac = ram_max_frac
        self.etl_cache = EtlCache()
        self.memory_window_size = memory_window_size
        self.execution_safety_margin = execution_safety_margin
        self.memory_histories = defaultdict(
            lambda: {
                "max_above_resting": deque(maxlen=self.memory_window_size),
                "model_size": deque(maxlen=self.memory_window_size),
            }
        )
        self.global_memory_histories = {
            "max_above_resting": deque(maxlen=self.memory_window_size),
            "model_size": deque(maxlen=self.memory_window_size),
        }
        self.cache_key_to_source_model = dict()
        self.pending_load_baselines = dict()

    def _source_model_key_from_model(self, model):
        base_model = getattr(getattr(model, "config", None), "base_model", None)
        if base_model is None:
            return None
        return getattr(base_model, "__name__", str(base_model))

    def _append_memory_sample(self, stat_name, source_model_key, value):
        if value is None:
            return
        value = max(int(value), 0)
        self.global_memory_histories[stat_name].append(value)
        if source_model_key is not None:
            self.memory_histories[source_model_key][stat_name].append(value)

    def _record_prediction_memory_sample(
        self, source_model_key, max_above_resting=None, model_size=None
    ):
        self._append_memory_sample(
            "max_above_resting", source_model_key, max_above_resting
        )
        self._append_memory_sample("model_size", source_model_key, model_size)
        estimated = self._estimated_memory_stats(source_model_key)
        self.max_above_resting = estimated["max_above_resting"]
        self.max_model_size = estimated["model_size"]

    def _estimated_memory_stats(
        self, source_model_key=None, allow_global_fallback=True
    ):
        stats = {}
        for stat_name in ("max_above_resting", "model_size"):
            history = None
            if source_model_key is not None:
                history = self.memory_histories[source_model_key][stat_name]
            if history:
                stats[stat_name] = max(history)
            elif allow_global_fallback and self.global_memory_histories[stat_name]:
                stats[stat_name] = max(self.global_memory_histories[stat_name])
            else:
                stats[stat_name] = 0
        return stats

    def _reset_peak_memory_stats(self):
        if not is_gpu_available():
            return
        reset_memory_stats = getattr(tf.config.experimental, "reset_memory_stats", None)
        if reset_memory_stats is None:
            return
        try:
            reset_memory_stats("GPU:0")
        except Exception:
            LOGGER.debug("Failed to reset TensorFlow peak memory stats.", exc_info=True)

    def _record_prediction_memory(self, model, resolved_cache_key):
        source_model_key = self._source_model_key_from_model(model)
        if is_gpu_available():
            in_use = BytesInUse()
            peak = MaxBytesInUse()
            max_above_resting = max(peak - in_use, 0)
        else:
            in_use = 0
            max_above_resting = 0

        load_baseline = self.pending_load_baselines.pop(resolved_cache_key, None)
        model_size = None
        if load_baseline is not None:
            model_size = max(in_use - load_baseline, 0)

        self._record_prediction_memory_sample(
            source_model_key,
            max_above_resting=max_above_resting,
            model_size=model_size,
        )

    def _has_memory_for_model(
        self,
        source_model_key=None,
        include_model_size=True,
        allow_global_fallback=True,
        extra_model_buffer=0,
    ):
        if self.gpu_memory_limit is None:
            return True  # first run
        estimated = self._estimated_memory_stats(
            source_model_key, allow_global_fallback=allow_global_fallback
        )
        self.max_above_resting = estimated["max_above_resting"]
        self.max_model_size = estimated["model_size"]
        if is_gpu_available():
            in_use = BytesInUse()
        else:
            LOGGER.info("No GPU available, skipping GPU memory checks.")
            self.max_above_resting = 0
            self.max_model_size = 0
            in_use = 0

        cpu_percent = psutil.virtual_memory().percent
        LOGGER.info(
            (
                "models loaded: {num_models}, in_use: {in_use}, max_above_resting: {mar},"
                " max_model_size: {mms}, gpu_memory_limit: {mem_limit}, cpu percent used: {cpu_percent}"
            ).format(
                num_models=len(self.loaded_models),
                in_use=bytes_to_meg(in_use),
                mar=bytes_to_meg(self.max_above_resting),
                mms=bytes_to_meg(self.max_model_size),
                mem_limit=bytes_to_meg(self.gpu_memory_limit),
                cpu_percent=cpu_percent,
            )
        )
        if cpu_percent > self.ram_max_frac * 100:
            return False
        required_gpu = (
            in_use + self.max_above_resting + self.reserved + extra_model_buffer
        )
        if include_model_size:
            required_gpu += self.max_model_size
        return required_gpu < self.gpu_memory_limit

    def _memory_for_one_more(self, source_model_key=None):
        return self._has_memory_for_model(source_model_key=source_model_key)

    def _ensure_cpu_headroom(self, exclude=None):
        exclude = set(exclude or [])
        while psutil.virtual_memory().percent > self.ram_max_frac * 100:
            if not self._close_oldest_model(exclude=exclude):
                return False
        return True

    def _ensure_prediction_headroom(
        self,
        source_model_key,
        active_cache_key,
        include_model_size,
        allow_global_fallback,
    ):
        estimated = self._estimated_memory_stats(
            source_model_key, allow_global_fallback=allow_global_fallback
        )
        global_model_size = 0
        if self.global_memory_histories["model_size"]:
            global_model_size = max(self.global_memory_histories["model_size"])
        transient_buffer = int(estimated["max_above_resting"] * 0.2)
        execution_buffer = max(
            global_model_size, self.execution_safety_margin, transient_buffer
        )
        while not self._has_memory_for_model(
            source_model_key=source_model_key,
            include_model_size=include_model_size,
            allow_global_fallback=allow_global_fallback,
            extra_model_buffer=execution_buffer,
        ):
            if not self._close_oldest_model(exclude={active_cache_key}):
                return False
            include_model_size = False
        if (
            self.gpu_memory_limit is not None
            and estimated["max_above_resting"] > self.gpu_memory_limit * 0.5
        ):
            while len(self.loaded_models) > 1:
                if not self._close_oldest_model(exclude={active_cache_key}):
                    return False
        return True

    def _close_oldest_model(self, exclude=None):
        exclude = set(exclude or [])
        for idx, name in enumerate(self.loaded_models):
            if name in exclude:
                continue
            self.loaded_models.pop(idx)
            self.model_cache[name].close(update_saver=False)
            del self.model_cache[name]
            self.pending_load_baselines.pop(name, None)
            self.cache_key_to_source_model.pop(name, None)
            gc.collect()
            return True
        LOGGER.info("No models cached -- cannot remove oldest model.")
        return False

    def model_cache_key(self, model, key, cache_key):
        if cache_key is None:
            if not isinstance(model, str):
                raise ValueError(
                    "To schedule a model with a file handle or BytesIO model you must provide a cache_key"
                )
            cache_key = f"model={model}"

        if key is None:
            return cache_key
        else:
            return f"{cache_key}_key={key}"

    def _rotate_in_model(self, model, key, config_overrides=None, cache_key=None):
        resolved_cache_key = self.model_cache_key(model, key=key, cache_key=cache_key)
        source_model_key = self.cache_key_to_source_model.get(resolved_cache_key)
        cache_miss = resolved_cache_key not in self.loaded_models
        if cache_miss:
            while (
                self.max_models is not None
                and len(self.loaded_models) + 1 > self.max_models
            ):
                if not self._close_oldest_model():
                    break
            self._ensure_cpu_headroom()
            config_overrides = config_overrides or {}
            merged_config = {**self.config, **config_overrides}
            load_baseline = BytesInUse() if is_gpu_available() else 0
            out_model = BaseModel.load(model, key=key, **merged_config)
            self.model_cache[resolved_cache_key] = out_model
            self.pending_load_baselines[resolved_cache_key] = load_baseline
            self.cache_key_to_source_model[
                resolved_cache_key
            ] = self._source_model_key_from_model(out_model)
            source_model_key = self.cache_key_to_source_model[resolved_cache_key]
        else:
            out_model = self.model_cache[resolved_cache_key]
            self.loaded_models.remove(
                resolved_cache_key
            )  # put it back at the end of the queue

        self.loaded_models.append(resolved_cache_key)
        self._ensure_prediction_headroom(
            source_model_key=source_model_key,
            active_cache_key=resolved_cache_key,
            include_model_size=cache_miss,
            allow_global_fallback=False,
        )
        out_model._cached_predict = True

        return out_model

    def _update_memory_limit(self, model):
        if hasattr(model.saver, "variables"):
            del model.saver.variables
            del model.saver.fallback_
        if is_gpu_available():
            gpu_memory_limit = BytesLimit()
            gpu_fraction = self.config.get("per_process_gpu_memory_fraction")
            if gpu_fraction is not None:
                gpu_memory_limit = min(
                    gpu_memory_limit, int(gpu_memory_limit * gpu_fraction)
                )
            self.gpu_memory_limit = gpu_memory_limit
        else:
            LOGGER.info("No GPU available, skipping GPU memory limit update.")
            self.gpu_memory_limit = sys.maxsize

    def close_all(self):
        while self.loaded_models:
            self._close_oldest_model()
        self.pending_load_baselines.clear()
        MODEL_REGISTRY.cleanup()

    @scheduled
    def predict(self, model_file, x, *args, key=None, model=None, **kwargs):
        return model.predict(x, *args, **kwargs)

    @scheduled
    def predict_proba(self, model_file, x, *args, key=None, model=None, **kwargs):
        return model.predict_proba(x, *args, **kwargs)

    @scheduled
    def attention_weights(self, model_file, x, *args, key=None, model=None, **kwargs):
        return model.attention_weights(x, *args, **kwargs)

    @scheduled
    def featurize(self, model_file, x, *args, key=None, model=None, **kwargs):
        return model.featurize(x, *args, **kwargs)

    @scheduled
    def featurize_sequence(self, model_file, x, *args, key=None, model=None, **kwargs):
        return model.featurize_sequence(x, *args, **kwargs)

    def in_cache(self, model, cache_key, key):
        resolved_cache_key = self.model_cache_key(model, key=key, cache_key=cache_key)
        return resolved_cache_key in self.loaded_models

    def etl_in_cache(self, model, cache_key):
        resolved_cache_key = self.model_cache_key(model, "etl", cache_key)
        return resolved_cache_key in self.etl_cache

    def load_etl(self, model_file_path, cache_key):
        resolved_cache_key = self.model_cache_key(model_file_path, "etl", cache_key)
        if resolved_cache_key in self.etl_cache:
            etl = self.etl_cache.get(resolved_cache_key)
        else:
            etl = SequenceLabeler.load(model_file_path, key="etl")
            self.etl_cache[resolved_cache_key] = etl
        return etl

    def get_model(self, model_file, key=None, config_overrides=None, cache_key=None):
        return self._rotate_in_model(
            model_file, key=key, config_overrides=config_overrides, cache_key=cache_key
        )


class EtlCache:
    def __init__(self, maxsize=128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key, None)

    def __contains__(self, key):
        return key in self.cache

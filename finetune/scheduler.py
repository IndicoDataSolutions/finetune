from collections import OrderedDict
import gc
import logging
import functools

import psutil
from finetune.target_models.sequence_labeling import SequenceLabeler

import tensorflow as tf
from finetune.base import BaseModel
from finetune.custom_ops import BytesInUse, BytesLimit, MaxBytesInUse
from finetune.errors import FinetuneSchedulerError

LOGGER = logging.getLogger("finetune")


def bytes_to_meg(x):
    return x / 1024 / 1024


def scheduled(fn):
    @functools.wraps(fn)
    def scheduled_predict(
        self, model_file, x, *args, key=None, config_overrides=None, **kwargs
    ):
        # this is just for backwards compat, should always have a blob key going forward
        cache_key = kwargs.pop("cache_key", None)
        model = self._rotate_in_model(
            model_file, key=key, config_overrides=config_overrides, cache_key=cache_key
        )
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
                preds = fn(
                    self, model_file=model_file, x=x, *args, model=model, **kwargs
                )
            except Exception as e:
                raise FinetuneSchedulerError(
                    "Original Error: {}, Retry Error: {}".format(
                        str(orig_except), str(e)
                    )
                )
        self._update_memory_limit(model)
        return preds

    return scheduled_predict


class Scheduler:
    def __init__(
        self, max_models=None, config=None, reserved=750000000, ram_max_frac=0.8
    ):
        self.loaded_models = list()
        self.max_models = max_models
        self.gpu_memory_limit = None
        self.model_cache = dict()
        self.max_above_resting = None
        self.previous_in_use = 0
        self.max_model_size = None
        self.config = config or {}
        self.reserved = reserved
        self.ram_max_frac = ram_max_frac
        self.etl_cache = EtlCache()

    def _memory_for_one_more(self):
        if self.gpu_memory_limit is None:
            return True  # first run

        in_use = BytesInUse()
        peak = MaxBytesInUse()
        if self.max_above_resting is None or (peak - in_use) > self.max_above_resting:
            self.max_above_resting = peak - in_use

        if (
            self.max_model_size is None
            or (in_use - self.previous_in_use) > self.max_model_size
        ):
            self.max_model_size = in_use - self.previous_in_use

        self.previous_in_use = in_use
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
        return (
            in_use + self.max_above_resting + self.max_model_size + self.reserved
        ) < self.gpu_memory_limit

    def _close_oldest_model(self):
        if len(self.loaded_models):
            name = self.loaded_models.pop(0)
            self.model_cache[name].close()
            del self.model_cache[name]
            gc.collect()
        else:
            LOGGER.info("No models cached -- cannot remove oldest model.")

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
        if resolved_cache_key not in self.loaded_models:
            if (
                self.max_models is not None
                and len(self.loaded_models) + 1 > self.max_models
            ) or not self._memory_for_one_more():
                self._close_oldest_model()
            config_overrides = config_overrides or {}
            merged_config = {**self.config, **config_overrides}
            out_model = BaseModel.load(model, key=key, **merged_config)
            self.model_cache[resolved_cache_key] = out_model
        else:
            out_model = self.model_cache[resolved_cache_key]
            self.loaded_models.remove(resolved_cache_key)  # put it back at the end of the queue

        self.loaded_models.append(resolved_cache_key)
        out_model._cached_predict = True

        return out_model

    def _update_memory_limit(self, model):
        if hasattr(model.saver, "variables"):
            del model.saver.variables
            del model.saver.fallback_
        self.gpu_memory_limit = (
            BytesLimit()
        )  # delay this so that any options get applied from finetune.

    def close_all(self):
        while self.loaded_models:
            self._close_oldest_model()

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

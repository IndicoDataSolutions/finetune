import gc
import logging

import tensorflow as tf
from finetune.base import BaseModel
from finetune.custom_ops import BytesInUse, BytesLimit, MaxBytesInUse

LOGGER = logging.getLogger("finetune")

def bytes_to_meg(x):
    return x / 1024 / 1024

class Scheduler:
    def __init__(self, max_models=None, config=None, reserved=750000000):
        self.loaded_models = list() 
        self.max_models = max_models
        self.gpu_memory_limit = None
        self.model_cache = dict()
        self.max_above_resting = None
        self.previous_in_use = 0
        self.max_model_size = None
        self.config = config or {}
        self.reserved = reserved

    def _memory_for_one_more(self):
        if self.gpu_memory_limit is None:
            return True # first run
        in_use = BytesInUse()
        peak = MaxBytesInUse()
        if self.max_above_resting is None or (peak - in_use) > self.max_above_resting:
            self.max_above_resting = peak - in_use

        if self.max_model_size is None or (in_use - self.previous_in_use) > self.max_model_size:
            self.max_model_size = in_use - self.previous_in_use

        self.previous_in_use = in_use
        LOGGER.info(
            (
                "models loaded: {num_models}, in_use: {in_use}, max_above_resting: {mar},"
                " max_model_size: {mms}, gpu_memory_limit: {mem_limit}"
            ).format(
                num_models=len(self.loaded_models),
                in_use=bytes_to_meg(in_use),
                mar=bytes_to_meg(self.max_above_resting),
                mms=bytes_to_meg(self.max_model_size),
                mem_limit=bytes_to_meg(self.gpu_memory_limit)
            )
        )
        return (in_use + self.max_above_resting + self.max_model_size + self.reserved) < self.gpu_memory_limit

    def _rotate_in_model(self, model):
        if model not in self.loaded_models:
            if (
                    self.max_models is not None and len(self.loaded_models) + 1 > self.max_models or
                    not self._memory_for_one_more()
            ):
                name = self.loaded_models.pop(0)
                self.model_cache[name].close()
                del self.model_cache[name]
                gc.collect()
            out_model = BaseModel.load(model, **self.config)
            self.model_cache[model] = out_model
        else:
            out_model = self.model_cache[model]
            self.loaded_models.remove(model) # put it back at the end of the queue
            
        self.loaded_models.append(model)
        out_model._cached_predict = True
        
        return out_model

    def _update_memory_limit(self, model):
        if hasattr(model.saver, "variables"):
            del model.saver.variables
            del model.saver.fallback_
        self.gpu_memory_limit = BytesLimit() # delay this so that any options get applied from finetune.

    def predict(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        predictions = model.predict(x, *args, **kwargs)
        self._update_memory_limit(model)
        return predictions

    def predict_proba(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        probas = model.predict_proba(x, *args, **kwargs)
        self._update_memory_limit(model)
        return probas

    def attention_weights(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        attn_weights = model.attention_weights(x, *args, **kwargs)
        self._update_memory_limit(model)
        return attn_weights

    def featurize(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        features = model.featurize(x, *args, **kwargs)
        self._update_memory_limit(model)
        return features

    def featurize_sequence(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        seq_features = model.featurize_sequence(x, *args, **kwargs)
        self._update_memory_limit(model)
        return seq_features

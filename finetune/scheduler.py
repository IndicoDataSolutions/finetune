import tensorflow as tf
from finetune import Classifier

class Scheduler:
    def __init__(self, max_models=None):
        self.loaded_models = list() # model queue
        self.max_models = max_models
        self.session = None
        self.gpu_memory_limit = None
        self.in_use_op = tf.contrib.memory_stats.BytesInUse()
        self.peak_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        self.model_cache = dict()
        self.max_above_resting = None
        self.previous_in_use = 0
        self.max_model_size = None
        
    def _memory_for_one_more(self):
        if self.session is None:
            return True # First prediction run?
        in_use, peak = self.session.run((self.in_use_op, self.peak_mem_op))
        if self.max_above_resting is None or (peak - in_use) > self.max_above_resting:
            self.max_above_resting = peak - in_use

        if self.max_model_size is None or (in_use - self.previous_in_use) > self.max_model_size:
            self.max_model_size = in_use - self.previous_in_use

        self.previous_in_use = in_use
        
        return (in_use + self.max_above_resting + self.max_model_size) < self.gpu_memory_limit

    def _rotate_in_model(self, model):
        if model not in self.loaded_models:
            if (
                    self.max_models is not None and len(self.loaded_models) + 1 > self.max_models_per_gpu or
                    not self._memory_for_one_more()
            ):
                name = self.loaded_models.pop(0)
                self.model_cache[name].close()
                del self.model_cache[name]
            out_model = Classifier.load(model) # doesn't matter its classifier                                                                        
            self.model_cache[model] = out_model

        else:
            out_model = self.model_cache[model]
            self.loaded_models.remove(model) # put it back at the end of the queue
            
        self.loaded_models.append(model)
        out_model._cached_predict = True
        
        return out_model

    def _update_memory_limit(self):
        if self.session is None:
            self.session = tf.Session() # delay this so that any options get applied from finetune.
            self.gpu_memory_limit = self.session.run(tf.contrib.memory_stats.BytesLimit())

    def predict(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        predictions = model.predict(x, *args, **kwargs)
        self._update_memory_limit()
        return predictions

    def predict_proba(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        probas = model.predict_proba(x, *args, **kwargs)
        self._update_memory_limit()
        return probas

    def attention_weights(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        attn_weights = model.attention_weights(x, *args, **kwargs)
        self._update_memory_limit()
        return attn_weights

    def featurize(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        attn_weights = model.featurize(x, *args, **kwargs)
        self._update_memory_limit()
        return attn_weights

    def featurize_sequence(self, model_file, x, *args, **kwargs):
        model = self._rotate_in_model(model_file)
        seq_features = model.featurize(x, *args, **kwargs)
        self._update_memory_limit()
        return seq_features

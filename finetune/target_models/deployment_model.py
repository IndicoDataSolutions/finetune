import numpy as np
import tqdm
import math
import itertools
import logging
import pandas as pd
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.data import Dataset

from finetune.saver import InitializeHook
from finetune.base import BaseModel
from finetune.target_models.comparison import ComparisonPipeline
from finetune.target_models.sequence_labeling import SequencePipeline, SequenceLabeler
from finetune.target_models.classifier import ClassificationPipeline
from finetune.base_models.bert.featurizer import bert_featurizer
from finetune.base_models.gpt.featurizer import gpt_featurizer
from finetune.base_models.gpt2.featurizer import gpt2_featurizer
from finetune.model import get_separate_model_fns, PredictMode
from finetune.errors import FinetuneError
from finetune.input_pipeline import BasePipeline

LOGGER = logging.getLogger("finetune")
PredictHook = namedtuple("InitializeHook", "feat_hook target_hook")


class TaskMode:
    SEQUENCE_LABELING = "Sequence_Labeling"
    CLASSIFICATION = "Classification"
    COMPARISON = "Comparison"


class DeploymentPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        self.pipeline_type = None
        self.pipeline = None

    def _target_encoder(self):
        raise NotImplementedError

    def _dataset_without_targets(self, Xs, train):
        if not callable(Xs):
            Xs_fn = lambda: self.wrap_tqdm(Xs, train)
        else:
            Xs_fn = lambda: self.wrap_tqdm(Xs(), train)

        dataset_encoded = lambda: itertools.chain.from_iterable(
            map(self.get_text_token_mask, Xs_fn())
        )
        if not callable(Xs) and self.config.chunk_long_sequences:
            # Adjust dataset size to account for long documents being chunked
            dataset_encoded_list = list(dataset_encoded())
            self.config.dataset_size = len(dataset_encoded_list)
        else:
            _ = (
                self.get_active_pipeline()
            )  # Call of dataset_encoded will refresh self.pipeline; if it is not called, still need to refresh

        types, _ = self.feed_shape_type_def()

        return Dataset.from_generator(dataset_encoded, output_types=types[0])

    def get_active_pipeline(self):
        pipelines = {
            "Classification": ClassificationPipeline,
            "Comparison": ComparisonPipeline,
            "Sequence_Labeling": SequencePipeline,
        }
        self.pipeline_type = pipelines[self.task]
        if (
            type(self.pipeline) != self.pipeline_type
        ):  # to prevent instantiating the same type of pipeline repeatedly
            if self.pipeline_type == SequencePipeline:
                self.pipeline = self.pipeline_type(
                    self.config, multi_label=self.multi_label
                )
            else:
                self.pipeline = self.pipeline_type(self.config)
        return self.pipeline_type

    def get_text_token_mask(self, *X):
        _ = self.get_active_pipeline()
        return self.pipeline.text_to_tokens_mask(*X)

    def get_shapes(self):
        _ = self.get_active_pipeline()
        _, shapes = self.pipeline.feed_shape_type_def()
        if hasattr(self, "dataset"):
            self.dataset.output_shapes = shapes[0]
        return shapes[0]

    def get_target_input_fn(self, features, batch_size=None):
        batch_size = batch_size or self.config.batch_size
        features = pd.DataFrame(features).to_dict("list")
        for key in features:
            features[key] = np.array(features[key])
        return tf.estimator.inputs.numpy_input_fn(
            features, batch_size=batch_size, shuffle=False
        )


class DeploymentModel(BaseModel):
    """ 
    Implements inference in arbitrary tasks in a cached manner by loading weights efficiently, allowing for quick interchanging of
    weights while avoiding slow graph recompilation.

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def __init__(self, featurizer=None, **kwargs):
        """
        For a full list of configuration options, see `finetune.config`.

        :param base_model: One of the base models from finetune.base_models, excluding textcnn.
        :param **kwargs: key-value pairs of config items to override.
        """
        self.input_pipeline = None

        if "base_model" in kwargs:
            if featurizer and kwargs["base_model"] != featurizer:
                raise FinetuneError("Base model passed in config does not match featurizer argument")
            featurizer = kwargs.pop("base_model")

        super().__init__(base_model=featurizer, **kwargs)
        if featurizer.featurizer not in [gpt2_featurizer, gpt_featurizer, bert_featurizer]:
            raise FinetuneError("Selected base model not supported.")
        self.validate_config()
        self.task = TaskMode.CLASSIFICATION
        self.input_pipeline.task = self.task
        self.featurizer_loaded = False
        self.adapters = False
        self.loaded_custom_previously = False

    def load_featurizer(self):
        """
        Performs graph compilation of the featurizer, saving most compilation overhead from occurring at predict time. Should
        be called after initialization but BEFORE any calls to load_custom_model or predict.
        """
        if not self.featurizer_loaded:
            self.featurizer_est = self._get_estimator("featurizer")
            self.predict_hooks.feat_hook.model_portion = "whole_featurizer"
            for hook in self.predict_hooks:
                hook.need_to_refresh = True
            self.predict(["finetune"], exclude_target=True)  # run arbitrary predict call to compile featurizer graph
            self.featurizer_loaded = True

    def load_custom_model(self, path):
        """
        Load in target model, and either adapters or entire featurizer from file. Must be called after load_featurizer.
        """
        if not self.featurizer_loaded:
            raise FinetuneError("Need to call load_featurizer before loading weights from file.")

        original_model = self.saver.load(path)
        original_model._trained = True

        if original_model.config.base_model.featurizer != self.config.base_model.featurizer:
            raise FinetuneError("You cannot mix featurizer types in the same deployment model instance.")

        if original_model.config.adapter_size != self.config.adapter_size:
            raise FinetuneError("adapter_size in config is compatible with this model")
        if type(self.config.base_model) != type(original_model.config.base_model):
            raise FinetuneError("Loaded file has incompatible base model.")
        if original_model.config.max_length != self.config.max_length:
            raise FinetuneError(
                "Loaded model has a different config.max_length than current value."
                " Changing max_length between loads is not yet supported."
            )

        self.adapters = original_model.config.adapter_size is not None

        self._target_model = original_model._target_model
        self._predict_op = original_model._predict_op
        self._predict_proba_op = original_model._predict_proba_op
        self._to_pull = 0
        self.loaded_custom_previously = True
        self._update_pipeline(original_model)
        self.load_featurizer()
        for hook in self.predict_hooks:
            hook.need_to_refresh = True
        if original_model.config.adapter_size is None:
            LOGGER.warning(
                "Loading without adapters will result in slightly slower load time than models that use adapters, and will also slow the next switch to an adapter model."
            )
            self.predict_hooks.feat_hook.model_portion = (
                "whole_featurizer"
            )  # need to load everything from save file, rather than standard base model file

        if (
            not self.adapters or not self.loaded_custom_previously
        ):  # previous model did not use adapters, so we have to update everything
            self.predict_hooks.feat_hook.refresh_base_model = True

    def _update_pipeline(self, original_model):
        """
        Refresh necessary attributes of DeploymentModel's input_pipeline so that it can support a newly loaded model
        """
        self.input_pipeline.target_dim = original_model.input_pipeline.target_dim
        self.input_pipeline.label_encoder = original_model.input_pipeline.label_encoder
        self.input_pipeline.text_encoder = original_model.input_pipeline.text_encoder
        self.input_pipeline._target_encoder = (
            original_model.input_pipeline._target_encoder
        )
        self.input_pipeline._post_data_initialization = (
            original_model.input_pipeline._post_data_initialization
        )
        self.input_pipeline._format_for_inference = (
            original_model.input_pipeline._format_for_inference
        )
        self.input_pipeline._format_for_encoding = (
            original_model.input_pipeline._format_for_encoding
        )
        self.config.chunk_long_sequences = original_model.config.chunk_long_sequences
        self.input_pipeline.config.chunk_long_sequences = original_model.config.chunk_long_sequences
        self.config.add_eos_bos_to_chunk = original_model.config.add_eos_bos_to_chunk
        self.input_pipeline.config.add_eos_bos_to_chunk = original_model.config.add_eos_bos_to_chunk

        if isinstance(original_model.input_pipeline, SequencePipeline):
            self.task = TaskMode.SEQUENCE_LABELING
            self.input_pipeline.multi_label = original_model.input_pipeline.multi_label
            self.multi_label = original_model.config.multi_label_sequences
            original_model.multi_label = self.multi_label
            self._initialize = original_model._initialize
        elif isinstance(original_model.input_pipeline, ComparisonPipeline):
            self.task = TaskMode.COMPARISON
        elif isinstance(original_model.input_pipeline, BasePipeline):
            self.task = TaskMode.CLASSIFICATION
        else:
            raise FinetuneError("Invalid pipeline in loaded file.")
        self.input_pipeline.task = self.task

    def _get_estimator(self, portion):
        assert portion in [
            "featurizer",
            "target",
        ], "Can only split model into featurizer and target."
        config = self._get_estimator_config()

        fn = get_separate_model_fns(
            target_model_fn=self._target_model if portion == "target" else None,
            predict_op=self._predict_op,
            predict_proba_op=self._predict_proba_op,
            build_target_model=self.input_pipeline.target_dim is not None,
            encoder=self.input_pipeline.text_encoder,
            target_dim=self.input_pipeline.target_dim if portion == "target" else None,
            label_encoder=self.input_pipeline.label_encoder if portion == "target" else None,
            saver=self.saver,
            portion=portion,
            build_attn=not isinstance(self.input_pipeline, ComparisonPipeline),
        )

        estimator = tf.estimator.Estimator(
            model_dir=self.estimator_dir, model_fn=fn, config=config, params=self.config
        )

        if hasattr(self, "predict_hooks") and portion == "featurizer":
            for hook in self.predict_hooks:
                hook.need_to_refresh = True
        elif not hasattr(self, "predict_hooks"):
            feat_hook = InitializeHook(self.saver, model_portion="featurizer")
            target_hook = InitializeHook(self.saver, model_portion="target")
            self.predict_hooks = PredictHook(feat_hook, target_hook)
        return estimator

    def _get_input_pipeline(self):
        if self.input_pipeline is None:
            self.input_pipeline = DeploymentPipeline(self.config)
        return self.input_pipeline

    def featurize(self, X):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        features = self.predict(X, exclude_targets=True)
        return features["features"]

    def _get_input_fn(self, gen):
        return self.input_pipeline.get_predict_input_fn(gen)

    def _inference(
        self,
        Xs,
        predict_keys=[PredictMode.NORMAL],
        exclude_target=False,
        n_examples=None,
    ):
        Xs = self.input_pipeline._format_for_inference(Xs)
        self._data = Xs
        self._closed = False
        n = n_examples or len(self._data)
        if self.adapters:
            self.predict_hooks.feat_hook.model_portion = "featurizer"
        else:
            self.predict_hooks.feat_hook.model_portion = "whole_featurizer"

        if self._predictions is None:
            featurizer_est = self._get_estimator("featurizer")
            self._predictions = featurizer_est.predict(
                input_fn=self._get_input_fn(self._data_generator),
                predict_keys=None,
                hooks=[self.predict_hooks.feat_hook],
                yield_single_examples=False,
            )

        self._clear_prediction_queue()

        num_batches = math.ceil(n / self.config.batch_size)
        features = [None] * n
        for i in tqdm.tqdm(range(num_batches), total=num_batches, desc="Featurization by Batch"):
            y = next(self._predictions)
            for j in range(
                self.config.batch_size
            ):  # this loop needed since yield_single_examples is False. In this case, n = # of predictions * batch_size
                single_example = {key: value[j] for key, value in y.items()}
                if self.config.batch_size * i + j > n - 1:
                    #  this is a result of the generator using cached_example and to_pull. If this is the last batch,
                    #  we need to check that all examples come from self._data and are not cached examples
                    break
                features[self.config.batch_size * i + j] = single_example

        if exclude_target:  # to initialize featurizer weights in load_featurizer
            return features

        preds = None
        if features is not None:
            self.predict_hooks.target_hook.need_to_refresh = True
            target_est = self._get_estimator("target")
            target_fn = self.input_pipeline.get_target_input_fn(features)
            preds = target_est.predict(
                input_fn=target_fn,
                predict_keys=predict_keys,
                hooks=[self.predict_hooks.target_hook],
            )

        predictions = [None] * n

        for i in tqdm.tqdm(range(n), total=n, desc="Target Model"):
            y = next(preds)
            try:
                y = y[predict_keys[0]] if len(predict_keys) == 1 else y
            except ValueError:
                raise FinetuneError(
                    "Cannot call `predict()` on a model that has not been fit."
                )
            predictions[i] = y

        self._clear_prediction_queue()
        return predictions

    def predict(self, X, exclude_target=False):
        """
        Performs inference using the weights and targets from the model in filepath used for load_custom_model. 

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        if self.task == TaskMode.SEQUENCE_LABELING and not exclude_target:
            return SequenceLabeler.predict(self, X)
        else:
            raw_preds = self._inference(X, exclude_target=exclude_target)
            if exclude_target:
                return raw_preds
            return self.input_pipeline.label_encoder.inverse_transform(
                np.asarray(raw_preds)
            )

    def predict_proba(self, X):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return super().predict_proba(X)

    def finetune(self, X, Y=None, batch_size=None):
        raise NotImplementedError

    @classmethod
    def get_eval_fn(cls):
        raise NotImplementedError

    @staticmethod
    def _target_model(
        config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs
    ):
        raise NotImplementedError

    def _predict_op(self, logits, **kwargs):
        raise NotImplementedError

    def _predict_proba_op(self, logits, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def attention_weights(self, Xs):
        raise NotImplementedError

    def generate_text(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def create_base_model(self, *args, **kwargs):
        raise NotImplementedError

    def cached_predict(self):
        raise NotImplementedError

    def load(cls, path, **kwargs):
        raise NotImplementedError


    @classmethod
    def finetune_grid_search(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def finetune_grid_search_cv(cls, *args, **kwargs):
        raise NotImplementedError

import logging

import tensorflow as tf
from finetune.base import BaseModel
from finetune.input_pipeline import BasePipeline
from finetune.errors import FinetuneError

from finetune.target_models.sequence_labeling import SequenceLabeler
from finetune.encoding.sequence_encoder import indico_to_finetune_sequence

LOGGER = logging.getLogger('finetune')


def get_input_fns(task_id, input_fn, validation_fn):
    def fn(x, y):
        return (
            {
                "tokens": reshape_to_rank_4(x["tokens"]),
                "mask": x["mask"],
                "task_id": task_id
            },
            y
        )

    return lambda: input_fn().map(fn), lambda: validation_fn().map(fn)


def reshape_to_rank_4(t):
    s = tf.shape(t)
    return tf.reshape(t, [s[0], -1, s[-2], s[-1]])


def get_train_eval_dataset(input_fn, val_size):
    def fn():
        return input_fn().take(val_size)

    return fn


class MultiTaskPipeline(BasePipeline):

    def __init__(self, *args, **kwargs):
        super(MultiTaskPipeline, self).__init__(*args, **kwargs)
        self.dataset_size_ = 0
        self.loss_weights = None
        self.target_dim = -1
        self.input_pipelines = None

    @property
    def dataset_size(self):
        return self.dataset_size_

    def get_train_input_fns(self, Xs, Y=None, batch_size=None, val_size=None):
        val_funcs = {}
        val_sizes = {}
        val_intervals = {}
        input_pipelines = {}
        frequencies = []
        input_funcs = []

        for task_name in self.config.tasks:
            input_pipelines[task_name] = self.config.tasks[task_name]._get_input_pipeline(self)
            task_tuple = input_pipelines[task_name].get_train_input_fns(
                Xs[task_name],
                Y[task_name],
                batch_size=batch_size,
                val_size=val_size
            )
            self.dataset_size_ += self.config.dataset_size
            frequencies.append(self.config.dataset_size)

            (val_func, input_func, val_sizes[task_name], val_intervals[task_name]) = task_tuple
            task_id = self.config.task_name_to_id[task_name]

            input_func_normalised, val_func_normalised = get_input_fns(task_id, input_func, val_func)
            input_funcs.append(input_func_normalised)
            val_funcs[task_name] = val_func_normalised
            val_funcs[task_name + "_train"] = get_train_eval_dataset(input_func_normalised, val_sizes[task_name])

        sum_frequencies = sum(frequencies)
        weights = [float(w) / sum_frequencies for w in frequencies]
        train_dataset = lambda: tf.data.experimental.sample_from_datasets([f() for f in input_funcs], weights)

        self.config.task_input_pipelines = input_pipelines
        self.config.dataset_size = self.dataset_size_
        return val_funcs, train_dataset, val_sizes, val_intervals

    def _target_encoder(self):
        raise FinetuneError("This should never be used??")


def get_loss_logits_fn(task, featurizer_state, config, targets_i, train, reuse, task_id_i):
    def loss_logits():
        with tf.variable_scope("target_model_{}".format(task)):
            target_model_out = config.tasks[task]._target_model(
                config=config,
                featurizer_state=featurizer_state,
                targets=targets_i,
                n_outputs=config.task_input_pipelines[task].target_dim,
                train=train,
                reuse=reuse
            )
            logits = target_model_out["logits"]
            logits.set_shape(None)
            return target_model_out["losses"], logits

    return tf.equal(task_id_i, config.task_name_to_id[task]), loss_logits


class MultiTask(BaseModel):
    """
    Target model for multi task learning. The approach used is to sample mini-batches from each task proportional to
    the size of the task for each dataset.

    :param tasks: A dictionary of pairs mapping string task names to model classes.
        eg. `{"sst": Classifier, "ner": SequenceLabeler}`
    :param \**kwargs: key-value pairs of config items to override. Note: The same config is used for each base task.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.config.tasks = tasks
        self.config.task_name_to_id = dict(zip(self.config.tasks.keys(), range(len(self.config.tasks))))

    def _get_input_pipeline(self):
        return MultiTaskPipeline(self.config)

    def cached_predict(self):
        """
        Context manager that prevents the recreation of the tensorflow graph on every call to BaseModel.predict().

        Not supported for MultiTask.
        """
        raise FinetuneError("cached_predict is not supported yet for MTL")

    def featurize(self, X):
        """                                                                                                                                                                        
        Runs featurization on the trained model for any of the tasks the model was trained for. Input and output formats  
        are the same as for each of the individial tasks.

        :param X: A dictionary mapping from task name to data, in the format required by the task type.
        :return: A dictionary mapping from task name to the features for that task.
        """
        features = {}
        for name, ModelClass in self.config.tasks.items():
            if name not in X:
                continue
            pred_model = ModelClass()
            pred_model.config = self.config
            pred_model.input_pipeline = self.config.task_input_pipelines[name]
            pred_model._initialize()
            pred_model.saver.variables = {
                k.replace("/target_model_{}".format(name), ""): v for k, v in self.saver.variables.items()
            }
            features[name] = pred_model.featurize(X[name])
        return features

    def predict(self, X):
        """
        Runs inference on the trained model for any of the tasks the model was trained for. Input and output formats
        are the same as for each of the individial tasks.


        :param X: A dictionary mapping from task name to data, in the format required by the task type.
        :return: A dictionary mapping from task name to the predictions for that task.
        """
        predictions = {}
        for name, ModelClass in self.config.tasks.items():
            if name not in X:
                continue
            pred_model = ModelClass()
            pred_model.config = self.config
            pred_model.input_pipeline = self.config.task_input_pipelines[name]
            pred_model._initialize()
            pred_model.saver.variables = {
                k.replace("/target_model_{}".format(name), ""): v for k, v in self.saver.variables.items()
            }
            predictions[name] = pred_model.predict(X[name])
        return predictions

    def predict_proba(self, X):
        """
        Runs probability inference on the trained model for any of the tasks the model was trained for. Falls back
        to normal predict when probabilities are not available for a task, eg Regression.

        Input and output formats are the same as for each of the individial tasks.

        :param X: A dictionary mapping from task name to data, in the format required by the task type.
        :return: A dictionary mapping from task name to the predictions for that task.
        """
        predictions = {}
        for name, ModelClass in self.config.tasks.items():
            if name not in X:
                continue
            pred_model = ModelClass()
            pred_model.config = self.config
            pred_model.input_pipeline = self.config.task_input_pipelines[name]
            pred_model._initialize()
            pred_model.saver.variables = {
                k.replace("/target_model_{}".format(name), ""): v for k, v in self.saver.variables.items()
            }
            try:
                predictions[name] = pred_model.predict_proba(X[name])
            except FinetuneError as e:
                LOGGER.warning(
                    (
                        "Probabilities are not available for {} and failed with exception {}."
                        "Falling back to regular predictions for this task."
                    ).format(name, e)
                )
                predictions[name] = pred_model.predict(X[name])
        return predictions

    def finetune(self, X, Y=None, batch_size=None):
        """

        :param X: A dictionary mapping from task name to inputs in the same format required for each of the models.
        :param Y: A dictionary mapping from task name to targets in the same format required for each of the models.
        :param batch_size: Number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        :return:
        """
        for t in [task_name for task_name, t in self.config.tasks.items() if t == SequenceLabeler]:
            X[t], Y[t], *_ = indico_to_finetune_sequence(X[t], labels=Y[t], multi_label=False, none_value="<PAD>")
        return super().finetune(X, Y=Y, batch_size=batch_size)

    @staticmethod
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, task_id=None, **kwargs):
        pred_fn_pairs = []
        featurizer_state["features"] = tf.cond(
            tf.equal(tf.shape(featurizer_state["features"])[1], 1),
            true_fn=lambda: tf.squeeze(featurizer_state["features"], [1]),
            false_fn=lambda: featurizer_state["features"]
        )

        targets_i = targets

        for task in config.tasks:
            pred, loss_logits = get_loss_logits_fn(task, featurizer_state, config, targets_i, train, reuse, task_id)
            pred_fn_pairs.append((pred, loss_logits))

        losses_logits = tf.case(
            pred_fn_pairs,
            default=None,
            exclusive=True,
            strict=True,
            name='top_selection'
        )

        return {
            "logits": losses_logits[1],
            "losses": losses_logits[0]
        }

    def _predict_op(self, logits, **kwargs):
        return tf.no_op()

    def _predict_proba_op(self, logits, **kwargs):
        return tf.no_op()

import tensorflow as tf
from finetune.base import BaseModel
from finetune.input_pipeline import BasePipeline
from finetune.errors import FinetuneError

from finetune.sequence_labeling import SequenceLabeler
from finetune.utils import indico_to_finetune_sequence


def get_input_fns(tid, i_fn, v_fn):
    def fn(x, y):
        return (
            {"tokens": reshape_to_rank_4(x["tokens"]), "mask": x["mask"]},
            {"target": y,"task_id": tid}
        )

    return lambda: i_fn().map(fn), lambda: v_fn().map(fn)


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
            task_id =  self.config.task_name_to_id[task_name]
            
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


def get_loss_fn(task, featurizer_state, config, targets_i, train, reuse, task_id_i):
    def loss():
        with tf.variable_scope("target_model_{}".format(task)):
            with tf.control_dependencies([tf.print(tf.shape(featurizer_state["features"]))]):
                loss_tensor = config.tasks[task]._target_model(
                    config=config,
                    featurizer_state=featurizer_state,
                    targets=targets_i,
                    n_outputs=config.task_input_pipelines[task].target_dim,
                    train=train,
                    reuse=reuse
                )["losses"]

            return loss_tensor
    return tf.equal(task_id_i, config.task_name_to_id[task]), loss


class MultiTask(BaseModel):

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self._is_seq_task = [task_name for task_name, t in tasks.items() if t == SequenceLabeler]
        self.config.tasks = {task_name: t for task_name, t in tasks.items()}
        self.config.task_name_to_id = dict(zip(self.config.tasks.keys(), range(len(self.config.tasks))))
        print(self.config.task_name_to_id)

    def _get_input_pipeline(self):
        return MultiTaskPipeline(self.config)

    def featurize(self, X):
        return super().featurize(X)

    def predict(self, X):
        raise FinetuneError("Predict is not implemented yet for MTL")

    def predict_proba(self, X):
        raise FinetuneError("Predict is not implemented yet for MTL")

    def finetune(self, X, Y=None, batch_size=None):
        for t in self._is_seq_task:
            X[t], Y[t], *_ = indico_to_finetune_sequence(X[t], labels=Y[t], multi_label=False, none_value="<PAD>")
        return super().finetune(X, Y=Y, batch_size=batch_size)

    @staticmethod
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        pred_fn_pairs = []
        featurizer_state["features"] = tf.cond(
            tf.equal(tf.shape(featurizer_state["features"])[1], 1),
            true_fn=lambda: tf.squeeze(featurizer_state["features"], [1]),
            false_fn=lambda: featurizer_state["features"]
        )

        task_id_i = targets["task_id"]
        targets_i = targets["target"]
        
        for task in config.tasks:
            pred, loss = get_loss_fn(task, featurizer_state, config, targets_i, train, reuse, task_id_i)
            pred_fn_pairs.append((pred, loss))

        return {
            "logits": tf.no_op(),
            "losses": tf.case(
                pred_fn_pairs,
                default=None,
                exclusive=True,
                strict=True,
                name='top_selection'
            )
        }

    def _predict_op(self, logits, **kwargs):
        return tf.no_op()

    def _predict_proba_op(self, logits, **kwargs):
        return tf.no_op()

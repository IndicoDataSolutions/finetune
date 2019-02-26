import tensorflow as tf

from finetune.base import BaseModel
from finetune.input_pipeline import BasePipeline
from finetune.errors import FinetuneError

from finetune.sequence_labeling import SequenceLabeler
from finetune.utils import indico_to_finetune_sequence


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
        weights = []
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
            weights.append(self.config.dataset_size)
            (val_funcs[task_name], input_func, val_sizes[task_name], val_intervals[task_name]) = task_tuple

            input_funcs.append(
                lambda: input_func().map(
                    lambda x, y: (
                        x,
                        {
                            "target": y,
                            "task_id": self.config.task_name_to_id[task_name]
                        }
                    )
                )
            )

        sum_weights = sum(weights)
        train_dataset = lambda: tf.data.experimental.sample_from_datasets([f() for f in input_funcs], [float(w) / sum_weights for w in weights])
        self.config.task_input_pipelines = input_pipelines
        return val_funcs, train_dataset, val_sizes, val_intervals

    def _target_encoder(self):
        raise FinetuneError("This should never be used??")


class MultiTask(BaseModel):

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self._is_seq_task = [task_name for task_name, t in tasks.items() if t == SequenceLabeler]
        self.config.tasks = {task_name: t for task_name, t in tasks.items()}
        self.config.task_name_to_id = dict(zip(self.config.tasks.keys(), range(len(self.config.tasks))))

    def _get_input_pipeline(self):
        return MultiTaskPipeline(self.config)

    def featurize(self, X):
        return super().featurize(X)

    def predict(self, X):
        raise FinetuneError("Predict does not make sense for MTL")

    def predict_proba(self, X):
        raise FinetuneError("Predict does not make sense for MTL")

    def finetune(self, X, Y=None, batch_size=None):
        """
        :param X: list or array of text.
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        for t in self._is_seq_task:
            X[t], Y[t], *_ = indico_to_finetune_sequence(X[t], labels=Y[t], multi_label=False, none_value="<PAD>")
        return super().finetune(X, Y=Y, batch_size=batch_size)

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        pred_fn_pairs = []
        for task in self.config.tasks:
            task_id = targets["task_id"]
            targets = targets["target"]

            with tf.variable_scope("target_model_{}".format(task)):
                task_loss = self.config.tasks[task]._target_model(
                    featurizer_state=featurizer_state,
                    targets=targets,
                    n_outputs=self.config.task_input_pipelines[task].target_dim,
                    train=train,
                    reuse=reuse
                )["losses"]# * self.input_pipeline.loss_weights[task]

            

            pred_fn_pairs.append(
                (
                    tf.equal(task_id, self.config.task_name_to_id[task]),
                    lambda: task_loss
                )
            )

        return {
            "logits": tf.no_op(),
            "losses": tf.case(
                pred_fn_pairs,
                default=None,
                exclusive=False,
                strict=False,
                name='top_selection'
            )
        }

    def _predict_op(self, logits, **kwargs):
        return tf.no_op()

    def _predict_proba_op(self, logits, **kwargs):
        return tf.no_op()

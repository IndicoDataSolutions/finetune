import math
import logging

import tqdm
import tensorflow as tf
from tensorflow.python.training import training
from tensorflow.python.distribute import device_util
from tensorflow.contrib.distribute import ParameterServerStrategy
from tensorflow.contrib.distribute.python.parameter_server_strategy import ParameterServerExtended
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib

from finetune.errors import FinetuneError

LOGGER = logging.getLogger("finetune")
_LOCAL_CPU = "/device:CPU:0"
_LOCAL_GPU_0 = "/device:GPU:0"


class ProgressHook(training.SessionRunHook):

    def __init__(self, n_batches, n_epochs=None, mode='train'):
        if mode not in ('train', 'predict'):
            raise FinetuneError("Invalid value for `ProgressHook` mode: {}".format(mode))
        self.mode = mode
        self.iterations = 0
        self.n_epochs = n_epochs
        if self.n_epochs:
            self.batches_per_epoch = int(n_batches / n_epochs)
        else:
            self.batches_per_epoch = n_batches
        self.progress_bar = None

    def epoch_descr(self, current_epoch):
        return "Epoch {}/{}".format(current_epoch, self.n_epochs)

    def write_description(self, current_epoch):
        if self.mode == 'train':
            self.progress_bar.set_description(self.epoch_descr(current_epoch))
        else:
            self.progress_bar.set_description("Inference")

    def log_progress(self):
        self.iterations += 1
        current_epoch = self.iterations // self.batches_per_epoch + 1
        current_batch = self.iterations % self.batches_per_epoch

        if current_batch == 0 and current_epoch != 1:
            current_epoch -= 1
            current_batch = self.batches_per_epoch

        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(total=self.batches_per_epoch)

        self.write_description(current_epoch)

        self.progress_bar.n = current_batch
        self.progress_bar.refresh()

    def after_run(self, run_context, run_values):
        self.log_progress()

    def end(self, session):
        self.progress_bar.n = self.batches_per_epoch
        self.write_description(self.n_epochs)
        self.progress_bar.refresh()
        del self.progress_bar


class PatchedParameterServerExtended(ParameterServerExtended):
    """Implementation of ParameterServerStrategy."""

    def __init__(self, container_strategy, num_gpus_per_worker, visible_gpus):
        super(ParameterServerExtended, self).__init__(container_strategy)
        self._num_gpus_per_worker = num_gpus_per_worker
        self._visible_gpus = visible_gpus
        self._initialize_local(num_gpus_per_worker)

        # We typically don't need to do all-reduce in this strategy.
        self._cross_device_ops = (
            cross_device_ops_lib.ReductionToOneDeviceCrossDeviceOps(
                reduce_to_device=_LOCAL_CPU
            )
        )

    def _initialize_local(self, num_gpus_per_worker):
        """Initialize internal devices for local training."""
        self._worker_device = device_util.canonicalize("/device:CPU:0")
        # Define compute devices which is a list of device strings and one for each
        # replica. When there are GPUs, replicate operations on these GPUs.
        # Otherwise, place operations on CPU.
        if num_gpus_per_worker > 0:
            self._compute_devices = tuple(
                map("/device:GPU:{}".format, self._visible_gpus)
            )
        else:
            self._compute_devices = (_LOCAL_CPU,)

        self._compute_devices = tuple(
            map(device_util.resolve, self._compute_devices)
        )
        self._canonical_compute_device_set = set(self._compute_devices)

        # If there is only one GPU, put everything on that GPU. Otherwise, place
        # variables on CPU.
        if num_gpus_per_worker == 1:
            assert len(self._compute_devices) == 1
            self._variable_device = _LOCAL_GPU_0
            self._parameter_devices = (_LOCAL_GPU_0,)
        else:
            self._variable_device = _LOCAL_CPU
            self._parameter_devices = (_LOCAL_CPU,)

        self._is_chief = True
        self._cluster_spec = None
        self._task_type = None
        self._task_id = None


class PatchedParameterServerStrategy(ParameterServerStrategy):

    def __init__(self, visible_gpus):
        """Initializes this strategy.
        Args:
        visible_gpus: device IDs of local GPUs to use
        """
        super(ParameterServerStrategy, self).__init__(
            PatchedParameterServerExtended(
                self,
                num_gpus_per_worker=len(visible_gpus),
                visible_gpus=visible_gpus
            )
        )


class LazySummaryHook(tf.train.SummarySaverHook):
    def __init__(self, save_steps=None,
                 save_secs=None,
                 output_dir=None,
                 summary_writer=None):
        super().__init__(save_steps=save_steps, save_secs=save_secs, output_dir=output_dir,
                         summary_writer=summary_writer, scaffold=1)  # scaffold = 1 suppresses exception in __init__

    def _get_summary_op(self):
        """Fetches the summary op either from self._summary_op or self._scaffold.

        Returns:
          Returns a list of summary `Tensor`.
        """
        if self._summary_op is not None:
            summary_op = self._summary_op
        else:
            summary_op = tf.train.Scaffold.get_or_default('summary_op',
                                                           tf.GraphKeys.SUMMARY_OP,
                                                           tf.summary.merge_all)
        if not isinstance(summary_op, list):
            return [summary_op]
        return summary_op

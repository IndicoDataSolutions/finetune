import tensorflow as tf
from tensorflow.python.distribute import device_util
from tensorflow.contrib.distribute import ParameterServerStrategy
from tensorflow.contrib.distribute.python.parameter_server_strategy import ParameterServerExtended
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib


_LOCAL_CPU = "/device:CPU:0"
_LOCAL_GPU_0 = "/device:GPU:0"


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

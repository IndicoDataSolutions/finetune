import tensorflow as tf
from tensorflow.python.distribute import device_util
from tensorflow.contrib.distribute import ParameterServerStrategy
from tensorflow.contrib.distribute.python.parameter_server_strategy import ParameterServerExtended
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib


_LOCAL_CPU = "/device:CPU:0"
_LOCAL_GPU_0 = "/device:GPU:0"


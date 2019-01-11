import os
import logging

import tensorflow as tf

from finetune.tasks.models import *

__version__, VERSION, version = ("0.5.13",) * 3

# Logging configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('finetune')
LOGGER.setLevel(logging.INFO)

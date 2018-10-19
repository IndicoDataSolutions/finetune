import os
import logging

import tensorflow as tf

from finetune.multifield import MultifieldClassifier, MultifieldRegressor
from finetune.classifier import Classifier
from finetune.regressor import Regressor
from finetune.sequence_labeling import SequenceLabeler
from finetune.comparison import Comparison
from finetune.multi_label_classifier import MultiLabelClassifier
from finetune.multiple_choice import MultipleChoice

__version__, VERSION, version = ("0.5.2",) * 3


# Logging configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('finetune')
LOGGER.setLevel(logging.INFO)

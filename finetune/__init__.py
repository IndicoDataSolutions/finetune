import logging
import os

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except ImportError:
    pass

from tensorflow.compat.v1 import logging as tf_logging

from finetune.target_models.classifier import Classifier
from finetune.target_models.document_labeling import DocumentLabeler
from finetune.target_models.multi_label_classifier import MultiLabelClassifier
from finetune.target_models.sequence_labeling import SequenceLabeler

__version__, VERSION, version = ("1.0.0",) * 3


# Logging configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf_logging.set_verbosity(tf_logging.ERROR)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("finetune")
LOGGER.setLevel(logging.INFO)

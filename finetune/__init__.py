import os
import logging

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except ImportError:
    pass

import tensorflow as tf
from tensorflow.compat.v1 import logging as tf_logging

from finetune.target_models.multifield import MultiFieldClassifier, MultiFieldRegressor
from finetune.target_models.classifier import Classifier
from finetune.target_models.regressor import Regressor
from finetune.target_models.sequence_labeling import SequenceLabeler
from finetune.target_models.comparison import Comparison
from finetune.target_models.multi_label_classifier import MultiLabelClassifier
from finetune.target_models.multiple_choice import MultipleChoice
from finetune.target_models.comparison_regressor import ComparisonRegressor
from finetune.target_models.ordinal_regressor import (
    OrdinalRegressor,
    ComparisonOrdinalRegressor,
)
from finetune.target_models.language_model import LanguageModel
from finetune.target_models.masked_language_model import MaskedLanguageModel
from finetune.target_models.document_labeling import DocumentLabeler

__version__, VERSION, version = ("0.9.0",) * 3


# Logging configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf_logging.set_verbosity(tf_logging.ERROR)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("finetune")
LOGGER.setLevel(logging.INFO)

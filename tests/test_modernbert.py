import os
import tempfile

import pytest

from finetune import Classifier, SequenceLabeler
from finetune.base_models.modern_bert.model import ModernBertLargeModel, ModernBertModel
from finetune.scheduler import Scheduler

try:
    import torch
    from transformers import AutoTokenizer
    from transformers import ModernBertModel as TransformersModernBertModel

    modernbert_transformers_available = True
except ImportError:
    modernbert_transformers_available = False


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.mark.skipif(
    not modernbert_transformers_available,
    reason="Transformers ModernBert not installed",
)
@pytest.mark.parametrize(
    "finetune_model, huggingface_model, max_delta_threshold",
    [
        (ModernBertModel, "answerdotai/ModernBERT-base", 1e-4),
        (ModernBertLargeModel, "answerdotai/ModernBERT-large", 2e-4),
    ],
)
def test_feature_equivalence(finetune_model, huggingface_model, max_delta_threshold):
    text = "The quick brown fox jumps over the lazy dog"
    text2 = (
        "This is a long sentence that should trigger local attention and hopefully test that we are padding correctly. "
        * 15
    )
    batch_text = [text, text2]

    finetune_model = SequenceLabeler(
        base_model=finetune_model
    )  # , mixed_precision=False, float_16_predict=False)
    finetune_features = finetune_model.featurize_sequence(batch_text)

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
    model = TransformersModernBertModel.from_pretrained(
        huggingface_model, attn_implementation="eager", torch_dtype=torch.float32
    )

    with torch.no_grad():
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True)
        transformers_features = model(**inputs).last_hidden_state.to("cpu").numpy()

    for i in range(len(batch_text)):
        num_tokens = len(tokenizer.tokenize(batch_text[i]))
        deltas = finetune_features[i] - transformers_features[i][1 : num_tokens + 1]
        max_delta = deltas.max()
        assert (
            max_delta < max_delta_threshold
        ), f"Max delta between Finetune and Transformers is {max_delta} - expecting {max_delta_threshold}"
        deltas_above_threshold = deltas > 1e-5
        rate_above_threshold = deltas_above_threshold.sum() / (
            deltas.shape[0] * deltas.shape[1]
        )
        assert (
            rate_above_threshold < 0.01
        ), f"More than 1% of features have a delta above 1e-5: {rate_above_threshold}"

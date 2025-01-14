from transformers import AutoTokenizer, ModernBertModel
import torch
from finetune.base_models.modern_bert.model import ModernBertModel as FTModernBertModel
from finetune import SequenceLabeler
import numpy as np

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog"
    finetune_model = SequenceLabeler(base_model=FTModernBertModel)
    finetune_features = finetune_model.featurize_sequence([text])[0]

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="eager")

    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)

    with torch.no_grad():
        transformers_features = model(**inputs).last_hidden_state.to("cpu").numpy()[0][1:-1]
    
    print(finetune_features.shape == transformers_features.shape)
    print(np.abs(finetune_features - transformers_features))


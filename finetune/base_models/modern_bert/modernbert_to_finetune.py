
if __name__ == "__main__":
    import torch as nn
    import numpy as np
    import joblib as jl

    torch_model = nn.load("/Finetune/finetune/model/modern_bert/pytorch_model.bin")
    finetune_model = dict()
    mapping = {
        "model.embeddings.tok_embeddings.weight": ("model/featurizer/ModernBert/Embedding/EmbeddingWithPadIdx/embeddings:0", False),
        "model.embeddings.norm.weight": (f"model/featurizer/ModernBert/Embedding/EmbeddingNorm/gamma:0", False),
    }

    for layer_i in range(22):
        layer_name = "" if layer_i == 0 else f"_{layer_i}"
        mapping[f"model.layers.{layer_i}.attn_norm.weight"] = (f"model/featurizer/ModernBert/EncoderLayer{layer_name}/AttnNorm/gamma:0", False)
        mapping[f'model.layers.{layer_i}.attn.Wqkv.weight'] = (f'model/featurizer/ModernBert/EncoderLayer{layer_name}/Attention/Wqkv/kernel:0', True)
        mapping[f'model.layers.{layer_i}.attn.Wo.weight'] = (f'model/featurizer/ModernBert/EncoderLayer{layer_name}/Attention/Wo/kernel:0', True)
        mapping[f"model.layers.{layer_i}.mlp_norm.weight"] = (f"model/featurizer/ModernBert/EncoderLayer{layer_name}/MLPNorm/gamma:0", False)
        mapping[f"model.layers.{layer_i}.mlp.Wi.weight"] = (f"model/featurizer/ModernBert/EncoderLayer{layer_name}/GLU/Wi/kernel:0", True)
        mapping[f"model.layers.{layer_i}.mlp.Wo.weight"] = (f"model/featurizer/ModernBert/EncoderLayer{layer_name}/GLU/Wo/kernel:0", True)
    mapping["model.final_norm.weight"] = ("model/featurizer/ModernBert/FinalNorm/gamma:0", False)

    print(mapping)
    for k, v in torch_model.items():
        if k.startswith("head") or k.startswith("decoder"):
            print("Skipping", k)
            continue
        print(f"===== {k} {v.shape} =====")
        new_name, do_transpose = mapping[k]    
        in_numpy = v.cpu().numpy()
        if do_transpose:
            in_numpy = np.transpose(in_numpy)
        print(f"Output = {new_name}, {in_numpy.shape}")
        finetune_model[new_name] = in_numpy

    jl.dump(finetune_model,  "modern_bert.jl")

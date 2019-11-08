import torch 
import joblib
import json
import numpy as np

mapping = {
    'roberta-model-sm.jl': 'roberta.base',
    'roberta-model-lg.jl': 'roberta.large'
}

for local_file, pytorch_name in mapping.items():
    current_model = joblib.load('finetune/model/bert/{}'.format(local_file))
    roberta = torch.hub.load('pytorch/fairseq', pytorch_name)
    model = roberta._modules['model']
    current_model['model/masked-language-model/output_bias:0'] = np.hstack([
        np.zeros(4),
        model.decoder.lm_head._parameters['bias'].detach().numpy(),
    ])
    current_model['model/masked-language-model/dense/kernel:0'] = model.decoder.lm_head.dense._parameters['weight'].detach().numpy()
    current_model['model/masked-language-model/dense/bias:0'] = model.decoder.lm_head.dense._parameters['bias'].detach().numpy()
    current_model['model/masked-language-model/LayerNorm/g:0'] = model.decoder.lm_head.layer_norm._parameters['weight'].detach().numpy()
    current_model['model/masked-language-model/LayerNorm/b:0'] = model.decoder.lm_head.layer_norm._parameters['bias'].detach().numpy()
    current_model['model/featurizer/bert/embeddings/word_embeddings:0'] = np.vstack([
        current_model['model/featurizer/bert/embeddings/word_embeddings:0'][:4],
        model.decoder.sentence_encoder.embed_tokens._parameters['weight'].detach().numpy()
    ])
    vocab = json.load(open('finetune/model/bert/roberta-large-vocab.json'))
    sorted_vocab = [item[0] for item in sorted(vocab.items(), key=lambda item: item[1])]
    open('finetune/model/bert/roberta_vocab.txt', 'w').write("\n".join(sorted_vocab))
    local_file_base = local_file.rpartition('.')[0]
    joblib.dump(current_model, 'finetune/model/bert/{}-v2.jl'.format(local_file_base))

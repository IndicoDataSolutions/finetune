import h5py

mapping = {
    'distillbert.jl': '/home/m/Downloads/distilbert-base-uncased-tf_model.h5',
    'distilroberta.jl': '/home/m/Downloads/distilroberta-base-tf_model.h5',
}

for local_file, h5_name in mapping.items():
    current_model = joblib.load('finetune/model/bert/{}'.format(local_file))
    h5 = htpy.File(h5_name)
    current_model['model/masked-language-model/output_bias:0'] = model.decoder.lm_head._parameters['weight']  
    current_model['model/masked-language-model/dense/kernel:0'] = model.decoder.lm_head.dense._parameters['weight']
    current_model['model/masked-language-model/dense/bias:0'] = model.decoder.lm_head.dense._parameters['bias']
    current_model['model/masked-language-model/LayerNorm/g:0'] = model.decoder.lm_head.layer_norm._parameters['weight']
    current_model['model/masked-language-model/LayerNorm/b:0'] = model.decoder.lm_head.layer_norm._parameters['bias']
    local_file_base = local_file.rpartition('.')[0]
    joblib.dump(current_model, 'finetune/model/bert/{}-v2.jl'.format(local_file_base))

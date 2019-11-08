import joblib
import ipdb

model_files = [
    'bert_large_cased.jl',
    'bert_small_cased.jl',
    'bert_small_multi_cased.jl',
    'bert_wwm_large_cased.jl'
]

for model_file in model_files:
    print("Processing ", model_file)
    model = joblib.load('finetune/model/bert/{}'.format(model_file))
    model_file_base = model_file.rpartition('.')[0]
    model['model/masked-language-model/output_bias:0'] = model.pop('model/featurizer/cls/predictions/output_bias:0')
    model['model/masked-language-model/dense/kernel:0'] = model.pop('model/featurizer/cls/predictions/transform/dense/kernel:0')
    model['model/masked-language-model/dense/bias:0'] = model.pop('model/featurizer/cls/predictions/transform/dense/bias:0')
    model['model/masked-language-model/LayerNorm/g:0'] = model.pop('model/featurizer/cls/predictions/transform/LayerNorm/gamma:0')
    model['model/masked-language-model/LayerNorm/b:0'] = model.pop('model/featurizer/cls/predictions/transform/LayerNorm/beta:0')
    joblib.dump(model, 'finetune/model/bert/{}-v2.jl'.format(model_file_base))
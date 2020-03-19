import numpy as np
import tensorflow as tf

class BiasModel(object):
    def fit(self, X, y, context=None):
        raise NotImplementedError

    def get_log_probas(self, X, context=None):
        raise NotImplementedError

def anonymize_headers(X):
    new_X = []
    for text in X:
        new_text = ' '.join([token if token.isdigit() else 'A' * len(token) for token in text.split()])
        new_X.append(new_text)
    return new_X

class TextBiasModel(BiasModel):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        X = anonymize_headers(X)
        self.model.fit(X, y)
    
    def get_log_probas(self, X):
        probs = [results[5] for results in self.model.process_long_sequence(X)]
        # import ipdb; ipdb.set_trace()
        log_probs = [np.log(ps) for ps in probs]
        max_length = max([len(ps) for ps in probs])
        return np.array([np.append(log_ps, [[0, 0]] * (max_length - len(log_ps)), axis=0)
                                    if len(log_ps) < max_length else log_ps for log_ps in log_probs])
        # return np.log(np.array(probs))
        # return tf.convert_to_tensor(np.log(np.array(probs)))

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
        probs = [results[6] for results in self.process_long_sequence(X)]
        return tf.convert_to_tensor(np.log(np.array(probs)))
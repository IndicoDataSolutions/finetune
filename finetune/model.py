import os
import random
import json
import warnings
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from functools import partial
from finetune.encoding import TextEncoder
from finetune.optimizers import AdamWeightDecay
from finetune.utils import find_trainable_variables, shape_list, assign_to_gpu, average_grads, iter_data, soft_split
from finetune.config import MAX_LENGTH, BATCH_SIZE, WEIGHT_STDDEV, N_EPOCHS, CLF_P_DROP, SEED, N_GPUS, WEIGHT_STDDEV, EMBED_P_DROP, RESID_P_DROP, N_HEADS, N_LAYER, ATTN_P_DROP, ACT_FN, LM_LOSS_COEF, LR, B1, B2, L2_REG, VECTOR_L2, EPSILON,LR_SCHEDULE, MAX_GRAD_NORM, LM_LOSS_COEF, LR_WARMUP
from finetune.train import block, dropout, embed, lr_schedules

SHAPES_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_shapes.json')
PARAM_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_{}.npy')
N_EMBED = 768


def clf(x, ny, w_init=tf.random_normal_initializer(stddev=WEIGHT_STDDEV), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b


def featurizer(X, M, encoder, train=False, reuse=None, max_length=MAX_LENGTH):
    with tf.variable_scope('model', reuse=reuse):
        embed_weights = tf.get_variable("we", [encoder.vocab_size + max_length, N_EMBED], initializer=tf.random_normal_initializer(stddev=WEIGHT_STDDEV))
        embed_weights = dropout(embed_weights, EMBED_P_DROP, train)

        X = tf.reshape(X, [-1, max_length, 2])
        M = tf.reshape(M, [-1, max_length])

        h = embed(X, embed_weights)
        for layer in range(N_LAYER):
            h = block(h, N_HEADS, ACT_FN, RESID_P_DROP, ATTN_P_DROP,'h%d'%layer, train=train, scale=True)

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h, [-1, N_EMBED]) # [batch * seq_len, embed]
        clf_token = encoder['_classify_']
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * max_length + pool_idx)

        clf_h = tf.reshape(clf_h, [-1, N_EMBED]) # [batch, embed]
        return {
            'embed_weights': embed_weights,
            'features': clf_h,
            'sequence_features': h
        }


def language_model(*, X, M, embed_weights, hidden, reuse=None):
    with tf.variable_scope('model', reuse=reuse):
        # language model ignores last hidden state because we don't have a target
        lm_h = tf.reshape(hidden[:, :-1], [-1, N_EMBED]) # [batch, seq_len, embed] --> [batch * seq_len, embed]
        lm_logits = tf.matmul(lm_h, embed_weights, transpose_b=True) # tied weights
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits,
            labels=tf.reshape(X[:, 1:, 0], [-1])
        )

        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
        lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)
        return {
            'logits': lm_logits,
            'losses': lm_losses,
        }


def classifier(hidden, targets, n_classes, train=False, reuse=None):
    with tf.variable_scope('model', reuse=reuse):
        hidden = dropout(hidden, CLF_P_DROP, train)
        clf_logits = clf(hidden, n_classes, train=train)
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=targets)
        return {
            'logits': clf_logits,
            'losses': clf_losses
        }


class LanguageModelClassifier(object):

    def __init__(self, max_length=MAX_LENGTH, *args, **kwargs):
        # ensure results are reproducible
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self._initialize()
        self.n_classes  = None

    def _initialize(self):
        self._set_random_seed(SEED)
        self.encoder = TextEncoder()

        # symbolic ops
        self.logits     = None # classification logits
        self.clf_loss   = None # cross-entropy loss
        self.lm_losses  = None # language modeling losses
        self.train      = None # gradient + parameter update
        self.features   = None # hidden representation fed to classifier
        self.is_built   = False
        self.is_trained = False

    def finetune(self, X, Y, batch_size=BATCH_SIZE):
        """
        X: List / array of text
        Y: Class labels
        """
        token_idxs = self.encoder.encode_for_classification(X, max_length=self.max_length)
        train_x, train_mask = self._array_format(token_idxs)
        n_batch_train = BATCH_SIZE * N_GPUS
        n_updates_total = (len(Y) // n_batch_train) * N_EPOCHS
        Y = self.label_encoder.fit_transform(Y)
        self.n_classes = len(self.label_encoder.classes_)
        self._build_model(n_updates_total=n_updates_total, n_classes=self.n_classes)

        dataset = shuffle(train_x, train_mask, Y, random_state=np.random)

        self.is_trained = True

        best_score = 0
        for i in range(N_EPOCHS):
            for xmb, mmb, ymb in iter_data(*dataset, n_batch=n_batch_train, verbose=True):
                cost, _ = self.sess.run([self.clf_loss, self.train_op], {self.X: xmb, self.M: mmb, self.Y: ymb})

        return self

    def fit(self, *args, **kwargs):
        # Alias for finetune
        return self.finetune(*args, **kwargs)

    def predict(self, X, max_length=None):
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.max_length
            for xmb, mmb in self._infer_prep(X, max_length=max_length):
                class_idx = self.sess.run(self.predict_op, {self.X: xmb, self.M: mmb})
                features = self.sess.run(self.features, {self.X: xmb, self.M: mmb})
                class_labels = self.label_encoder.inverse_transform(class_idx)
                predictions.append(class_labels)
        return np.concatenate(predictions)

    def predict_proba(self, X, max_length=None):
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.max_length
            for xmb, mmb in self._infer_prep(X, max_length=max_length):
                probas = self.sess.run(self.predict_proba_op, {self.X: xmb, self.M: mmb})
                classes = self.label_encoder.classes_
                predictions.extend([
                    dict(zip(classes, proba)) for proba in probas
                ])
        return np.asarray(predictions)

    def featurize(self, X, max_length=None):
        """
        Embed inputs in learned feature space
        TODO: enable featurization without finetuning (using pre-trained model only)
        """
        features = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.max_length
            for xmb, mmb in self._infer_prep(X, max_length=max_length):
                feature_batch = self.sess.run(self.features, {self.X: xmb, self.M: mmb})
                features.append(feature_batch)
        return np.concatenate(features)

    def transform(self, *args, **kwargs):
        return self.featurize(*args, **kwargs)

    def _infer_prep(self, X, max_length=None):
        max_length = max_length or self.max_length
        token_idxs = self.encoder.encode_for_classification(X, max_length=max_length)
        infer_x, infer_mask = self._array_format(token_idxs)
        n_batch_train = BATCH_SIZE * N_GPUS
        self._build_model(n_updates_total=0, n_classes=self.n_classes, reuse=self.is_built, train=False)
        yield from iter_data(infer_x, infer_mask, n_batch=n_batch_train, verbose=True)

    def _array_format(self, token_idxs):
        """
        Returns numpy array of token idxs and corresponding mask
        Returned `x` array contains two channels:
            0: byte-pair encoding embedding
            1: positional embedding
        """
        n = len(token_idxs)
        seq_lengths = [len(x) for x in token_idxs]
        x    = np.zeros((n, self.max_length, 2), dtype=np.int32)
        mask = np.zeros((n, self.max_length), dtype=np.float32)
        for i, seq_length in enumerate(seq_lengths):
            # BPE embedding
            x[i, :seq_length, 0] = token_idxs[i]
            # masking: value of 1 means "consider this in cross-entropy LM loss"
            mask[i, 1:seq_length] = 1
        # positional_embeddings
        x[:, :, 1] = np.arange(self.encoder.vocab_size, self.encoder.vocab_size + self.max_length)
        return x, mask

    def _compile_train_op(self, *, params, grads, n_updates_total):
        grads = average_grads(grads)
        grads = [grad for grad, param in grads]
        self.train_op = AdamWeightDecay(
            params=params,
            grads=grads,
            lr=LR,
            schedule=partial(lr_schedules[LR_SCHEDULE], warmup=LR_WARMUP),
            t_total=n_updates_total,
            l2=L2_REG,
            max_grad_norm=MAX_GRAD_NORM,
            vector_l2=VECTOR_L2,
            b1=B1,
            b2=B2,
            e=EPSILON
        )

    def _build_model(self, n_updates_total, n_classes, train=True, reuse=None):
        """
        Finetune language model on text inputs
        """
        gpu_ops = []
        gpu_grads = []
        self._define_placeholders()

        for i, (X, M, Y) in enumerate(soft_split(self.X, self.M, self.Y, n_splits=N_GPUS)):
            do_reuse = True if i > 0 else reuse
            device = tf.device(assign_to_gpu(i, "/gpu:0"))
            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)

            features_aggregator = []
            losses_aggregator = []
            with device, scope:
                featurizer_state = featurizer(X, M, encoder=self.encoder, train=train, reuse=do_reuse)
                language_model_state = language_model(
                    X=X,
                    M=M,
                    embed_weights=featurizer_state['embed_weights'],
                    hidden=featurizer_state['sequence_features'],
                    reuse=do_reuse
                )
                features_aggregator.append(featurizer_state['features'])

                if n_classes is not None:
                    classifier_state = classifier(
                        hidden=featurizer_state['features'],
                        targets=Y,
                        n_classes=n_classes,
                        train=train,
                        reuse=do_reuse
                    )
                    train_loss = tf.reduce_mean(classifier_state['losses'])

                    if LM_LOSS_COEF > 0:
                        train_loss += LM_LOSS_COEF * tf.reduce_mean(language_model_state['losses'])

                    params = find_trainable_variables("model")
                    grads = tf.gradients(train_loss, params)
                    grads = list(zip(grads, params))
                    gpu_grads.append(grads)
                    losses_aggregator.append([
                        classifier_state['logits'],
                        classifier_state['losses'],
                        language_model_state['losses']
                    ])

        self.features = tf.concat(features_aggregator, 0)

        if n_classes is not None:
            self.logits, self.clf_losses, self.lm_losses = [tf.concat(op, 0) for op in zip(*losses_aggregator)]
            self.predict_op = tf.argmax(self.logits, -1)
            self.predict_proba_op = tf.nn.softmax(self.logits, -1)
            self._compile_train_op(
                params=params,
                grads=gpu_grads,
                n_updates_total=n_updates_total
            )
            self.clf_loss = tf.reduce_mean(self.clf_losses)

        # Optionally load saved model
        if hasattr(self, '_save_path'):
            self._load_finetuned_model()
        elif not self.is_trained:
            self._load_base_model()

        self.is_built = True

    def _set_random_seed(self, seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def _define_placeholders(self):
        # tf placeholders
        self.X = tf.placeholder(tf.int32,   [None, self.max_length, 2]) # token idxs (BPE embedding + positional)
        self.M = tf.placeholder(tf.float32, [None, self.max_length])    # sequence mask
        self.Y = tf.placeholder(tf.int32,   [None])                     # classification targets

    def _load_base_model(self):
        """
        Load serialized base model parameters into tf Tensors
        """
        pretrained_params = find_trainable_variables('model', exclude='model/clf')
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.sess.run(tf.global_variables_initializer())

        shapes = json.load(open('model/params_shapes.json'))
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:self.max_length]
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(len(self.encoder.special_tokens), N_EMBED) * WEIGHT_STDDEV).astype(np.float32), init_params[0]], 0)
        del init_params[1]
        self.sess.run([p.assign(ip) for p, ip in zip(pretrained_params, init_params)])

    def __getstate__(self):
        """
        Leave serialization of all tf objects to tf
        """
        required_fields = ['label_encoder', 'max_length', 'n_classes', '_save_path']
        serialized_state = {
            k: v for k, v in self.__dict__.items()
            if k in required_fields
        }
        return serialized_state

    def save(self, path):
        """
        Save in two steps:
            - Serialize tf graph to disk using tf.Saver
            - Serialize python model using pickle

        Note:
            Does not serialize state of Adam optimizer.
            Should not be used to save / restore a training model.
        """

        # Setting self._save_path indicates that we should load the saved model from
        # disk at next training / inference
        self._save_path = path
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(self.sess, path)
        pickle.dump(self, open(self._save_path + '.pkl', 'wb'))
        del self._save_path

    @classmethod
    def load(cls, path):
        """
        Load in three steps:
            - Load pickled python object
            - Clear tf graph
            - Load serialized session using tf.train.Saver
        """
        if not path.endswith('.pkl'):
            path += '.pkl'
        model = pickle.load(open(path, 'rb'))
        model._initialize()
        tf.reset_default_graph()
        return model

    def _load_finetuned_model(self):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self._save_path)

        # if _save_path is present on the model, the saved model is loaded
        # from disk the next time predict / predict_proba / featurize is called
        del self._save_path
        self.is_trained = True

if __name__ == "__main__":
    headers = ['annotator', 'target', 'original_target', 'text']
    train_df = pd.read_csv(
        "data/cola.train.csv",
        names=headers,
        delimiter='\t'
    )
    validation_df = pd.read_csv(
        "data/cola.dev.csv",
        names=headers,
        delimiter='\t'
    )
    model = LanguageModelClassifier()
    model.finetune(train_df.text.values, train_df.target.values)
    model.save('saved-models/cola')

    predictions = model.predict(validation_df.text.values)
    true_labels = validation_df.target.values
    from sklearn.metrics import matthews_corrcoef
    mc = matthews_corrcoef(true_labels, predictions)
    print(mc)

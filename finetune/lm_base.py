import os
import random
import warnings
import logging
import pickle
import json

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

import tempfile

from functools import partial

from finetune.download import download_data_if_required
from finetune.encoding import TextEncoder
from finetune.optimizers import AdamWeightDecay, schedules
from sklearn.model_selection import train_test_split
from finetune.config import (
    MAX_LENGTH, BATCH_SIZE, N_EPOCHS, SEED, WEIGHT_STDDEV, LR, B1, B2, L2_REG, VECTOR_L2,
    EPSILON, LR_SCHEDULE, MAX_GRAD_NORM, LM_LOSS_COEF, LR_WARMUP, N_EMBED, ROLLING_AVG_DECAY
)

from finetune.target_encoders import OneHotLabelEncoder, RegressionEncoder
from finetune.network_modules import featurizer, language_model, classifier, regressor
from finetune.utils import find_trainable_variables, get_available_gpus, shape_list, assign_to_gpu, average_grads, \
    iter_data, soft_split

SHAPES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'params_shapes.json')
PARAM_PATH = os.path.join(os.path.dirname(__file__), 'model', 'params_{}.npy')

_LOGGER = logging.getLogger(__name__)

DROPOUT_ON = 1
DROPOUT_OFF = 0


class LanguageModelBase(object, metaclass=ABCMeta):
    """
    A sklearn-style class for finetuning a Transformer language model on a classification task.

    :param max_length: Determines the number of tokens to be included in the document representation.
                       Providing more than `max_length` tokens to the model as input will result in truncation.
    """

    def __init__(self, autosave_path=None, max_length=MAX_LENGTH, verbose=True):
        self.autosave_path = autosave_path or tempfile.mkdtemp()
        _LOGGER.info("Writing intermediate checkpoints to {}".format(self.autosave_path))
        self.max_length = max_length
        self.autosave_path = autosave_path
        self.label_encoder = None
        self._initialize()
        self.target_dim = None
        self._load_from_file = False
        self.verbose = verbose
        self.is_classification = None

    def _initialize(self):
        self._set_random_seed(SEED)

        download_data_if_required()
        self.encoder = TextEncoder()

        # symbolic ops
        self.logits = None  # classification logits
        self.clf_loss = None  # cross-entropy loss
        self.lm_losses = None  # language modeling losses
        self.train = None  # gradient + parameter update
        self.features = None  # hidden representation fed to classifier
        self.summaries = None  # Tensorboard summaries
        self.train_writer = None
        self.valid_writer = None

        # indicator vars
        self.is_built = False  # has tf graph been constructed?
        self.is_trained = False  # has model been fine-tuned?

    def _text_to_ids(self, *Xs, max_length=None):
        max_length = max_length or self.max_length
        assert len(Xs) == 1, "This implementation assumes a single Xs"
        token_idxs = self.encoder.encode_for_classification(Xs[0], max_length=max_length)
        tokens, mask = self._array_format(token_idxs)
        return tokens, mask

    def target_model(self, hidden, targets, n_outputs, train=False, reuse=None):
        if self.is_classification:
            return classifier(hidden, targets, n_outputs, self.do_dropout, train=train, reuse=reuse)
        return regressor(hidden, targets, n_outputs, self.do_dropout, train=train, reuse=reuse)

    def predict_ops(self, logits):
        if self.is_classification:
            return tf.argmax(logits, -1), tf.nn.softmax(logits, -1)
        return logits, logits

    def get_target_encoder(self):
        if self.is_classification:
            return OneHotLabelEncoder()
        else:
            return RegressionEncoder()

    def _finetune(self, *Xs, Y, batch_size=BATCH_SIZE, val_size=0.05, val_interval=150, val_window_size=5):
        """
        X: List / array of text
        Y: Class labels
        val_size: Float fraction or int number that represents the size of the validation set.
        val_interval: The interval for which validation is performed, measured in number of steps.
        """
        self.label_encoder = self.get_target_encoder()
        train_x, train_mask = self._text_to_ids(*Xs)
        n_batch_train = batch_size * max(len(get_available_gpus()), 1)
        n_updates_total = (len(Y) // n_batch_train) * N_EPOCHS
        Y = self.label_encoder.fit_transform(Y)
        self.target_dim = len(self.label_encoder.target_dim)
        self._build_model(n_updates_total=n_updates_total, target_dim=self.target_dim)

        dataset = shuffle(train_x, train_mask, Y, random_state=np.random)
        x_tr, x_va, m_tr, m_va, y_tr, y_va = train_test_split(*dataset, test_size=val_size, random_state=31415)

        dataset = (x_tr, m_tr, y_tr)
        val_dataset = (x_va, m_va, y_va)

        self.is_trained = True
        avg_train_loss = 0
        avg_val_loss = 0
        global_step = 0
        best_val_loss = float("inf")
        val_window = [float("inf")] * val_window_size
        for i in range(N_EPOCHS):
            for xmb, mmb, ymb in iter_data(*dataset, n_batch=n_batch_train, verbose=True):
                global_step += 1
                if global_step % val_interval == 0:

                    summary = self.sess.run([self.summaries], {self.X: xmb, self.M: mmb, self.Y: ymb, self.do_dropout: DROPOUT_OFF})
                    self.train_writer.add_summary(summary, global_step)

                    sum_val_loss = 0
                    for xval, mval, yval in iter_data(*val_dataset, n_batch=n_batch_train, verbose=True):
                        val_cost, summary = self.sess.run([self.clf_loss, self.summaries],
                                                          {self.X: xval, self.M: mval, self.Y: yval, self.do_dropout: DROPOUT_OFF})
                        self.valid_writer.add_summary(summary, global_step)
                        sum_val_loss += val_cost
                        avg_val_loss = avg_val_loss * ROLLING_AVG_DECAY + val_cost * (1 - ROLLING_AVG_DECAY)
                        _LOGGER.info("\nVAL: LOSS = {}, ROLLING AVG = {}".format(val_cost, avg_val_loss))
                    val_window.append(sum_val_loss)
                    val_window.pop(0)

                    if np.mean(val_window) <= best_val_loss:
                        best_val_loss = np.mean(val_window)
                        _LOGGER.info("Autosaving new best model.")
                        self.save(self.autosave_path)
                        _LOGGER.info("Done!!")
                cost, _ = self.sess.run([self.clf_loss, self.train_op], {self.X: xmb, self.M: mmb, self.Y: ymb, self.do_dropout: DROPOUT_ON})
                avg_train_loss = avg_train_loss * ROLLING_AVG_DECAY + cost * (1 - ROLLING_AVG_DECAY)
                _LOGGER.info("\nTRAIN: LOSS = {}, ROLLING AVG = {}".format(cost, avg_train_loss))

        return self

    @abstractmethod
    def finetune(self, *args, **kwargs):
        """
        """

    def fit(self, *args, **kwargs):
        """
        An alias for finetune.
        """
        return self.finetune(*args, **kwargs)

    def _predict(self, *Xs, max_length=None):
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.max_length
            for xmb, mmb in self._infer_prep(*Xs, max_length=max_length):
                class_idx = self.sess.run(self.predict_op, {self.X: xmb, self.M: mmb})
                class_labels = self.label_encoder.inverse_transform(class_idx)
                predictions.append(class_labels)
        return np.concatenate(predictions).tolist()

    @abstractmethod
    def predict(self, *args, **kwargs):
        """"""

    def _predict_proba(self, *Xs, max_length=None):
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.max_length
            for xmb, mmb in self._infer_prep(*Xs, max_length=max_length):
                probas = self.sess.run(self.predict_proba_op, {self.X: xmb, self.M: mmb})
                classes = self.label_encoder.target_dim
                predictions.extend([
                    dict(zip(classes, proba)) for proba in probas
                ])
        return predictions

    @abstractmethod
    def predict_proba(self, *args, **kwargs):
        """"""

    def _featurize(self, *Xs, max_length=None):
        features = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.max_length
            for xmb, mmb in self._infer_prep(*Xs, max_length=max_length):
                feature_batch = self.sess.run(self.features, {self.X: xmb, self.M: mmb})
                features.append(feature_batch)
        return np.concatenate(features)

    @abstractmethod
    def featurize(self, *args, **kwargs):
        """"""

    def transform(self, *args, **kwargs):
        """
        An alias for `featurize`.
        """
        return self.featurize(*args, **kwargs)

    def _infer_prep(self, *X, max_length=None):
        max_length = max_length or self.max_length
        infer_x, infer_mask = self._text_to_ids(*X, max_length=max_length)
        n_batch_train = BATCH_SIZE * max(len(get_available_gpus()), 1)
        self._build_model(n_updates_total=0, target_dim=self.target_dim, train=False)
        yield from iter_data(infer_x, infer_mask, n_batch=n_batch_train, verbose=self.verbose)

    def _array_format(self, token_idxs):
        """
        Returns numpy array of token idxs and corresponding mask
        Returned `x` array contains two channels:
            0: byte-pair encoding embedding
            1: positional embedding
        """
        n = len(token_idxs)
        seq_lengths = [len(x) for x in token_idxs]
        x = np.zeros((n, self.max_length, 2), dtype=np.int32)
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

        self.summaries += tf.contrib.training.add_gradients_summaries(grads)

        grads = [grad for grad, param in grads]
        self.train_op = AdamWeightDecay(
            params=params,
            grads=grads,
            lr=LR,
            schedule=partial(schedules[LR_SCHEDULE], warmup=LR_WARMUP),
            t_total=n_updates_total,
            l2=L2_REG,
            max_grad_norm=MAX_GRAD_NORM,
            vector_l2=VECTOR_L2,
            b1=B1,
            b2=B2,
            e=EPSILON
        )

    def _construct_graph(self, n_updates_total, target_dim, train=True):
        gpu_grads = []
        self.summaries = []

        # store whether or not graph was previously compiled with dropout
        self.train = train
        self._define_placeholders()

        features_aggregator = []
        losses_aggregator = []

        train_loss_tower = 0
        gpus = get_available_gpus()
        n_splits = max(len(gpus), 1)
        for i, (X, M, Y) in enumerate(soft_split(self.X, self.M, self.Y, n_splits=n_splits)):
            do_reuse = True if i > 0 else tf.AUTO_REUSE

            if gpus:
                device = tf.device(assign_to_gpu(gpus[i], params_device=gpus[0]))
            else:
                device = tf.device('cpu')

            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)

            with device, scope:
                featurizer_state = featurizer(X, encoder=self.encoder, dropout_placeholder=self.do_dropout, train=train,
                                              reuse=do_reuse)
                language_model_state = language_model(
                    X=X,
                    M=M,
                    embed_weights=featurizer_state['embed_weights'],
                    hidden=featurizer_state['sequence_features'],
                    reuse=do_reuse
                )
                features_aggregator.append(featurizer_state['features'])

                if target_dim is not None:
                    target_model_state = self.target_model(
                        hidden=featurizer_state['features'],
                        targets=Y,
                        n_outputs=target_dim,
                        train=train,
                        reuse=do_reuse
                    )
                    train_loss = tf.reduce_mean(target_model_state['losses'])

                    if LM_LOSS_COEF > 0:
                        train_loss += LM_LOSS_COEF * tf.reduce_mean(language_model_state['losses'])

                    train_loss_tower += train_loss

                    params = find_trainable_variables("model")
                    grads = tf.gradients(train_loss, params)
                    grads = list(zip(grads, params))
                    gpu_grads.append(grads)
                    losses_aggregator.append([
                        target_model_state['logits'],
                        target_model_state['losses'],
                        language_model_state['losses']
                    ])

        self.features = tf.concat(features_aggregator, 0)

        if target_dim is not None:
            self.logits, self.clf_losses, self.lm_losses = [tf.concat(op, 0) for op in zip(*losses_aggregator)]
            self.predict_op, self.predict_proba_op = self.predict_ops(self.logits)
            self._compile_train_op(
                params=params,
                grads=gpu_grads,
                n_updates_total=n_updates_total
            )
            self.clf_loss = tf.reduce_mean(self.clf_losses)
            self.summaries.append(tf.summary.scalar('TargetModelLoss', self.clf_loss))
            self.summaries.append(tf.summary.scalar('LanguageModelLoss', tf.reduce_mean(self.lm_losses)))
            self.summaries.append(tf.summary.scalar('TotalLoss', train_loss_tower / n_splits))
            self.summaries = tf.summary.merge(self.summaries)

    def _build_model(self, n_updates_total, target_dim, train=True):
        """
        Construct tensorflow symbolic graph.
        """
        if not self.is_trained or train != self.train:
            # reconstruct graph to include/remove dropout
            # #if `train` setting has changed
            self._construct_graph(n_updates_total, target_dim, train=train)

        # Optionally load saved model
        if self._load_from_file:
            self._load_finetuned_model()
        elif not self.is_trained:
            self._load_base_model()

        if train:
            self.train_writer = tf.summary.FileWriter(self.autosave_path + '/train', self.sess.graph)
            self.valid_writer = tf.summary.FileWriter(self.autosave_path + '/valid', self.sess.graph)
        self.is_built = True

    def _set_random_seed(self, seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def _define_placeholders(self):
        # tf placeholders
        self.X = tf.placeholder(tf.int32, [None, self.max_length, 2])  # token idxs (BPE embedding + positional)
        self.M = tf.placeholder(tf.float32, [None, self.max_length])  # sequence mask
        self.Y = tf.placeholder(tf.float32, [None, self.target_dim])  # classification targets
        self.do_dropout = tf.placeholder(tf.float32)  # 1 for do dropout and 0 to not do dropout

    def _load_base_model(self):
        """
        Load serialized base model parameters into tf Tensors
        """
        pretrained_params = find_trainable_variables('model', exclude='model/clf')
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.sess.run(tf.global_variables_initializer())

        with open(SHAPES_PATH) as shapes_file:
            shapes = json.load(shapes_file)
            offsets = np.cumsum([np.prod(shape) for shape in shapes])
            init_params = [np.load(PARAM_PATH.format(n)) for n in range(10)]
            init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
            init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
            init_params[0] = init_params[0][:self.max_length]
            special_embed = (np.random.randn(len(self.encoder.special_tokens), N_EMBED) * WEIGHT_STDDEV).astype(
                np.float32)
            init_params[0] = np.concatenate([init_params[1], special_embed, init_params[0]], 0)
            del init_params[1]
            self.sess.run([p.assign(ip) for p, ip in zip(pretrained_params, init_params)])

    def __getstate__(self):
        """
        Leave serialization of all tf objects to tf
        """
        required_fields = [
            'label_encoder', 'max_length', 'target_dim', '_load_from_file', 'verbose', 'autosave_path',
            'is_classification'
        ]
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

        # Setting self._load_from_file indicates that we should load the saved model from
        # disk at next training / inference. It is set temporarily so that the serialized
        # model includes this information.
        self._load_from_file = path
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(self.sess, path)
        pickle.dump(self, open(self._load_from_file + '.pkl', 'wb'))
        self._load_from_file = False

    @classmethod
    def load(cls, path):
        """
        Load a saved fine-tuned model from disk.

        :param path: string path name to load model from.  Same value as previously provided to :meth:`save`.
        """
        if not path.endswith('.pkl'):
            path += '.pkl'
        model = pickle.load(open(path, 'rb'))
        model._initialize()
        tf.reset_default_graph()
        return model

    def _load_finetuned_model(self):
        self.sess = tf.Session()
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess, self._load_from_file)
        self._load_from_file = False
        self.is_trained = True


class LanguageModelClassifier(LanguageModelBase):

    def featurize(self, X, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X, max_length=max_length)

    def predict(self, X, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return self._predict(X, max_length=max_length)

    def predict_proba(self, X, max_length=None):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        return self._predict_proba(X, max_length=max_length)

    def finetune(self, X, Y, batch_size=BATCH_SIZE, val_size=0.05, val_interval=150):
        """
        :param X: list or array of text.
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        :param val_size: Float fraction or int number that represents the size of the validation set.
        :param val_interval: The interval for which validation is performed, measured in number of steps.
        """
        return self._finetune(X, Y=Y, batch_size=batch_size, val_size=val_size, val_interval=val_interval)


class LanguageModelEntailment(LanguageModelBase):

    def __init__(self, *args, **vargs):
        super().__init__(*args, **vargs)
        self.label_encoder = OrdinalClassificationEncoder()

    def _text_to_ids(self, *Xs, max_length=None):
        max_length = max_length or self.max_length
        assert len(Xs) == 2, "This implementation assumes 2 Xs"

        question_answer_pairs = self.encoder.encode_for_entailment(*Xs, max_length=max_length)

        tokens, mask = self._array_format(question_answer_pairs)
        return tokens, mask

    def finetune(self, X_1, X_2, Y, batch_size=BATCH_SIZE, val_size=0.05, val_interval=150):
        """
        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param Y: integer or string-valued class labels. It is necessary for the items of Y to be sortable.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        :param val_size: Float fraction or int number that represents the size of the validation set.
        :param val_interval: The interval for which validation is performed, measured in number of steps.
        """
        return self._finetune(X_1, X_2, Y=Y, batch_size=batch_size, val_size=val_size, val_interval=val_interval)

    def predict(self, X_1, X_2, max_length=None):
        """
        Produces X_2 list of most likely class labels as determined by the fine-tuned model.

        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return self.label_encoder.inverse_transform(self._predict_proba(X_1, X_2, max_length=max_length))

    def predict_proba(self, X_1, X_2, max_length=None):
        """
        Produces X_2 probability distribution over classes for each example in X.

        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of dictionaries.  Each dictionary maps from X_2 class label to its assigned class probability.
        """
        return self._predict_proba(X_1, X_2, max_length=max_length)

    def featurize(self, X_1, X_2, max_length=None):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X_1: list or array of text to embed as the queries.
        :param X_2: list or array of text to embed as the answers.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X_1, X_2, max_length=max_length)

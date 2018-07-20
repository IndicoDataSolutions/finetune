import os
import random
import warnings
import logging
import pickle
import json

import tqdm

from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict
from functools import partial

import numpy as np
import tensorflow as tf

from finetune.download import download_data_if_required
from finetune.encoding import TextEncoder
from finetune.optimizers import AdamWeightDecay, schedules
from sklearn.model_selection import train_test_split

from finetune.utils import sample_with_temperature
from finetune.target_encoders import OneHotLabelEncoder, RegressionEncoder, SequenceLabelingEncoder
from finetune.network_modules import featurizer, language_model, classifier, regressor, sequence_labeler
from finetune.utils import find_trainable_variables, get_available_gpus, assign_to_gpu, average_grads, \
    iter_data, soft_split, sequence_decode, concat_or_stack
from finetune.errors import InvalidTargetType
from finetune.config import PAD_TOKEN, get_default_config

SHAPES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'params_shapes.json')
PARAM_PATH = os.path.join(os.path.dirname(__file__), 'model', 'params_{}.npy')

_LOGGER = logging.getLogger(__name__)

DROPOUT_ON = 1
DROPOUT_OFF = 0

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
SEQUENCE_LABELING = 'sequence-labeling'

SequenceArray = namedtuple("SequenceArray", ['token_ids', 'mask', 'labels'])


class BaseModel(object, metaclass=ABCMeta):
    """
    A sklearn-style class for finetuning a Transformer language model on a classification task.

    :param config: A config object, or None to use the default config.
    :param **kwargs: key-value pairs of config items to override.
    """

    def __init__(self, config=None, **kwargs):
        self.config = config or get_default_config()
        self.config.override_from_dict(kwargs)
        self.label_encoder = None
        self._initialize()
        self.target_dim = None
        self._load_from_file = False
        self.target_type = None

    def _initialize(self):
        # Initializes the non-serialized bits of the class.
        self._set_random_seed(self.config.seed)

        download_data_if_required()
        self.encoder = TextEncoder()

        # symbolic ops
        self.logits = None  # classification logits
        self.clf_loss = None  # cross-entropy loss
        self.lm_losses = None  # language modeling losses
        self.lm_predict_op = None
        self.train = None  # gradient + parameter update
        self.features = None  # hidden representation fed to classifier
        self.summaries = None  # Tensorboard summaries
        self.train_writer = None
        self.valid_writer = None
        self.predict_params = None

        # indicator vars
        self.is_built = False  # has tf graph been constructed?
        self.is_trained = False  # has model been fine-tuned?

    def _text_to_ids(self, *Xs, max_length=None):
        # Maps lists of text to formatted numpy arrays of token ids and loss-masks marking the lengths of the sequences.
        max_length = max_length or self.config.max_length
        assert len(Xs) == 1, "This implementation assumes a single Xs"
        token_idxs = self.encoder.encode_for_classification(Xs[0], max_length=max_length)
        seq_array = self._array_format(token_idxs)
        return seq_array.token_ids, seq_array.mask

    def _target_model(self, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        # Conditionally constructs the default model for each of the main ways in which finetune can be used.
        # Can be overridden to use a different target model.
        if self.target_type == CLASSIFICATION:
            return classifier(featurizer_state['features'], targets, n_outputs, self.do_dropout, config=self.config,
                              train=train, reuse=reuse, **kwargs)
        elif self.target_type == REGRESSION:
            return regressor(featurizer_state['features'], targets, n_outputs, self.do_dropout, config=self.config,
                             train=train, reuse=reuse, **kwargs)
        elif self.target_type == SEQUENCE_LABELING:
            return sequence_labeler(featurizer_state['sequence_features'], targets, n_outputs, self.do_dropout,
                                    config=self.config,
                                    train=train, reuse=reuse, **kwargs)
        else:
            raise InvalidTargetType(self.target_type)

    def _predict_ops(self, logits, **kwargs):
        # Gets the correct prediction methods for each of the main ways that the model can be used.
        if self.target_type == CLASSIFICATION:
            return tf.argmax(logits, -1), tf.nn.softmax(logits, -1)
        elif self.target_type == REGRESSION:
            return logits, tf.constant(0)  # TODO: Find something better than constant 0 for predict proba.
        elif self.target_type == SEQUENCE_LABELING:
            return sequence_decode(logits, kwargs.get("transition_matrix"))
        else:
            raise InvalidTargetType(self.target_type)

    def _get_target_encoder(self):
        # Gets the correct target encoder for the problem.
        if self.target_type == CLASSIFICATION:
            return OneHotLabelEncoder()
        elif self.target_type == REGRESSION:
            return RegressionEncoder()
        elif self.target_type == SEQUENCE_LABELING:
            return SequenceLabelingEncoder()
        else:
            raise InvalidTargetType(self.target_type)

    def _eval(self, *tensors, feed_dict):
        """
        Evaluate the value of each of the provided tensors.
        Returns a `dict` that maps from tensor to result value.  
        If any result value is None, that result is excluded from the results `dict`.
        """
        tensors = [
            tensor if tensor is not None else tf.no_op()
            for tensor in tensors
        ]
        values = self.sess.run(tensors, feed_dict=feed_dict)
        return {
            tensor: value
            for tensor, value in zip(tensors, values)
            if value is not None
        }

    def _finetune(self, *Xs, Y=None, batch_size=None):
        train_x, train_mask = self._text_to_ids(*Xs)
        return self._training_loop(train_x, train_mask, Y, batch_size)

    def _training_loop(self, train_x, train_mask, Y, batch_size=None):
        batch_size = batch_size or self.config.batch_size
        self.label_encoder = self._get_target_encoder()
        n_batch_train = batch_size * max(len(get_available_gpus(self.config)), 1)
        n_examples = train_x.shape[0]
        n_updates_total = (n_examples // n_batch_train) * self.config.n_epochs

        if Y is not None:
            Y = self.label_encoder.fit_transform(Y)
            target_dim = len(self.label_encoder.target_dim)
        else:
            # only language model will be trained, mock fake target
            Y = [[None]] * n_examples
            target_dim = None

        self._build_model(n_updates_total=n_updates_total, target_dim=target_dim)

        dataset = (train_x, train_mask, Y)

        x_tr, x_va, m_tr, m_va, y_tr, y_va = train_test_split(*dataset, test_size=self.config.val_size,
                                                              random_state=self.config.seed)
        dataset = (x_tr, m_tr, y_tr)
        val_dataset = (x_va, m_va, y_va)

        self.is_trained = True
        avg_train_loss = 0
        avg_val_loss = 0
        global_step = 0
        best_val_loss = float("inf")
        val_window = [float("inf")] * self.config.val_window_size
        for i in range(self.config.n_epochs):
            for xmb, mmb, ymb in iter_data(*dataset, n_batch=n_batch_train, verbose=self.config.verbose):
                global_step += 1
                if global_step % self.config.val_interval == 0:
                    tqdm.tqdm.write("Train loss is :{}, Val loss is :{}".format(avg_train_loss, avg_val_loss))

                    outputs = self._eval(
                        self.summaries,
                        feed_dict={
                            self.X: xmb,
                            self.M: mmb,
                            self.Y: ymb,
                            self.do_dropout: DROPOUT_OFF
                        }
                    )

                    if self.train_writer is not None:
                        self.train_writer.add_summary(outputs.get(self.summaries), global_step)

                    sum_val_loss = 0
                    for xval, mval, yval in iter_data(*val_dataset, n_batch=n_batch_train, verbose=self.config.verbose, tqdm_desc="Validation"):
                        outputs = self._eval(
                            self.clf_loss,
                            self.summaries,
                            feed_dict={
                                self.X: xval,
                                self.M: mval,
                                self.Y: yval,
                                self.do_dropout: DROPOUT_OFF
                            }
                        )

                        if self.valid_writer is not None:
                            self.valid_writer.add_summary(outputs.get(self.summaries), global_step)
                        val_cost = outputs.get(self.clf_loss, 0)
                        sum_val_loss += val_cost
                        avg_val_loss = (
                                avg_val_loss * self.config.rolling_avg_decay
                                + val_cost * (1 - self.config.rolling_avg_decay)
                        )
                    val_window.append(sum_val_loss)
                    val_window.pop(0)

                    if np.mean(val_window) <= best_val_loss:
                        best_val_loss = np.mean(val_window)
                        if self.config.save_best_model:
                            self.save(self.config.autosave_path)

                outputs = self._eval(
                    self.clf_loss, 
                    self.train_op,
                    feed_dict={
                      self.X: xmb,
                      self.M: mmb,
                      self.Y: ymb,
                      self.do_dropout: DROPOUT_ON
                    }
                )
                  
                cost = outputs.get(self.clf_loss, 0)
                avg_train_loss = avg_train_loss * self.config.rolling_avg_decay + cost * (
                        1 - self.config.rolling_avg_decay)

        return self

    @abstractmethod
    def finetune(self, *args, **kwargs):
        """ The base method for finetuning the model. """

    def fit(self, *args, **kwargs):
        """ An alias for finetune. """
        return self.finetune(*args, **kwargs)

    def _predict(self, *Xs, max_length=None):
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.config.max_length
            for xmb, mmb in self._infer_prep(*Xs, max_length=max_length):
                output = self._eval(self.predict_op,
                                    feed_dict={
                                        self.X: xmb,
                                        self.M: mmb,
                                        self.do_dropout: DROPOUT_OFF
                                    }
                                    )
                class_idx = output.get(self.predict_op)
                class_labels = self.label_encoder.inverse_transform(class_idx)
                predictions.append(class_labels)
        return np.concatenate(predictions).tolist()

    @abstractmethod
    def predict(self, *args, **kwargs):
        """ The base method for predicting from the model. """

    def _predict_proba(self, *Xs, max_length=None):
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.config.max_length
            for xmb, mmb in self._infer_prep(*Xs, max_length=max_length):
                output = self._eval(
                    self.predict_proba_op,
                    feed_dict={
                        self.X: xmb,
                        self.M: mmb,
                        self.do_dropout: DROPOUT_OFF
                    }
                )
                probas = output.get(self.predict_proba_op)
                classes = self.label_encoder.target_dim
                predictions.extend([
                    dict(zip(classes, proba)) for proba in probas
                ])
        return predictions

    @abstractmethod
    def predict_proba(self, *args, **kwargs):
        """ Base method for predicting, with probabilites. (when available) """

    def _featurize(self, *Xs, max_length=None):
        features = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.config.max_length
            for xmb, mmb in self._infer_prep(*Xs, max_length=max_length):
                feature_batch = self.sess.run(self.features, {self.X: xmb, self.M: mmb})
                features.append(feature_batch)
        return np.concatenate(features)

    @abstractmethod
    def featurize(self, *args, **kwargs):
        """
        Base method to get raw features out of the model.
        These features are the same that are fed into the target_model.
        """

    def transform(self, *args, **kwargs):
        """
        An alias for `featurize`.
        """
        return self.featurize(*args, **kwargs)

    def _infer_prep(self, *X, max_length=None):
        max_length = max_length or self.config.max_length
        infer_x, infer_mask = self._text_to_ids(*X, max_length=max_length)
        n_batch_train = self.config.batch_size * max(len(get_available_gpus(self.config)), 1)
        self._build_model(n_updates_total=0, target_dim=self.target_dim, train=False)
        yield from iter_data(infer_x, infer_mask, n_batch=n_batch_train, verbose=self.config.verbose)

    def _array_format(self, token_idxs, labels=None):
        """
        Returns numpy array of token idxs and corresponding mask
        Returned `x` array contains two channels:
            0: byte-pair encoding embedding
            1: positional embedding
        """
        n = len(token_idxs)
        seq_lengths = [len(x) for x in token_idxs]
        x = np.zeros((n, self.config.max_length, 2), dtype=np.int32)
        mask = np.zeros((n, self.config.max_length), dtype=np.float32)
        labels_arr = np.full((n, self.config.max_length), PAD_TOKEN, dtype='object') if labels else None
        for i, seq_length in enumerate(seq_lengths):
            # BPE embedding
            x[i, :seq_length, 0] = token_idxs[i]
            # masking: value of 1 means "consider this in cross-entropy LM loss"
            mask[i, 1:seq_length] = 1
            if labels:
                labels_arr[i, :seq_length] = labels[i]
        # positional_embeddings
        x[:, :, 1] = np.arange(self.encoder.vocab_size, self.encoder.vocab_size + self.config.max_length)
        return SequenceArray(
            token_ids=x,
            mask=mask,
            labels=labels_arr
        )

    def _compile_train_op(self, *, params, grads, n_updates_total):
        grads = average_grads(grads)

        if self.config.summarize_grads:
            self.summaries += tf.contrib.training.add_gradients_summaries(grads)

        grads = [grad for grad, param in grads]
        self.train_op = AdamWeightDecay(
            params=params,
            grads=grads,
            lr=self.config.lr,
            schedule=partial(schedules[self.config.lr_schedule], warmup=self.config.lr_warmup),
            t_total=n_updates_total,
            l2=self.config.l2_reg,
            max_grad_norm=self.config.max_grad_norm,
            vector_l2=self.config.vector_l2,
            b1=self.config.b1,
            b2=self.config.b2,
            e=self.config.epsilon
        )

    def _construct_graph(self, n_updates_total, target_dim=None, train=True):
        gpu_grads = []
        self.summaries = []

        # store whether or not graph was previously compiled with dropout
        self.train = train
        self.target_dim = target_dim
        self._define_placeholders()


        aggregator = defaultdict(list)
        train_loss_tower = 0
        gpus = get_available_gpus(self.config)
        n_splits = max(len(gpus), 1)
        for i, (X, M, Y) in enumerate(soft_split(self.X, self.M, self.Y, n_splits=n_splits)):
            do_reuse = True if i > 0 else tf.AUTO_REUSE

            if gpus:
                device = tf.device(assign_to_gpu(gpus[i], params_device=gpus[0]))
            else:
                device = tf.device('cpu')

            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)

            with device, scope:
                featurizer_state = featurizer(
                    X,
                    config=self.config,
                    encoder=self.encoder,
                    dropout_placeholder=self.do_dropout,
                    train=train,
                    reuse=do_reuse
                )
                language_model_state = language_model(
                    X=X,
                    M=M,
                    config=self.config,
                    embed_weights=featurizer_state['embed_weights'],
                    hidden=featurizer_state['sequence_features'],
                    reuse=do_reuse
                )

                lm_loss_coef = self.config.lm_loss_coef
                if target_dim is None:
                    lm_loss_coef = 1.0

                train_loss = lm_loss_coef * tf.reduce_mean(language_model_state['losses'])

                aggregator['features'].append(featurizer_state['features'])
                aggregator['lm_losses'].append(language_model_state['losses'])

                lm_logits = language_model_state["logits"]
                aggregator["lm_model"].append(sample_with_temperature(lm_logits, self.config.lm_temp))

                if target_dim is not None:
                    target_model_state = self._target_model(
                        featurizer_state=featurizer_state,
                        targets=Y,
                        n_outputs=target_dim,
                        train=train,
                        reuse=do_reuse,
                        max_length=self.config.max_length
                    )
                    train_loss += (1 - lm_loss_coef) * tf.reduce_mean(target_model_state['losses'])
                    train_loss_tower += train_loss

                    params = find_trainable_variables("model")
                    grads = tf.gradients(train_loss, params)
                    grads = list(zip(grads, params))
                    gpu_grads.append(grads)
                    aggregator['logits'].append(target_model_state['logits'])
                    aggregator['clf_losses'].append(target_model_state['losses'])

        self.lm_predict_op = tf.concat(aggregator["lm_model"], 0)
        self.features = tf.concat(aggregator['features'], axis=0)
        self.lm_losses = tf.concat(aggregator['lm_losses'], axis=0)

        if target_dim is not None:
            self.logits = tf.concat(aggregator['logits'], axis=0)
            self.clf_losses = concat_or_stack(aggregator['clf_losses'])

            self.predict_op, self.predict_proba_op = self._predict_ops(
                self.logits,
                **target_model_state.get("predict_params", {})
            )
            self._compile_train_op(
                params=params,
                grads=gpu_grads,
                n_updates_total=n_updates_total
            )
            self.clf_loss = tf.reduce_mean(self.clf_losses)
            self.lm_loss = tf.reduce_mean(self.lm_losses)
            self.summaries.append(tf.summary.scalar('TargetModelLoss', self.clf_loss))
            self.summaries.append(tf.summary.scalar('LanguageModelLoss', self.lm_loss))
            self.summaries.append(tf.summary.scalar('TotalLoss', train_loss_tower / n_splits))
            self.summaries = tf.summary.merge(self.summaries)

    def _build_model(self, n_updates_total, target_dim, train=True):
        """
        Construct tensorflow symbolic graph.
        """
        if not self.is_trained or train != self.train or self.target_dim != target_dim:
            # reconstruct graph to include/remove dropout
            # if `train` setting has changed
            self._construct_graph(n_updates_total, target_dim, train=train)

        # Optionally load saved model
        if self._load_from_file:
            self._load_finetuned_model()
        elif not self.is_trained:
            self._load_base_model()

        if train:
            if self.config.tensorboard_folder is not None:
                if not os.path.exists(self.config.tensorboard_folder):
                    os.mkdir(self.config.tensorboard_folder)
                self.train_writer = tf.summary.FileWriter(self.config.tensorboard_folder + '/train', self.sess.graph)
                self.valid_writer = tf.summary.FileWriter(self.config.tensorboard_folder + '/valid', self.sess.graph)
        self.is_built = True

    def _initialize_session(self):
        gpus = get_available_gpus(self.config)
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(gpu) for gpu in gpus])
        conf = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=conf)

    def _set_random_seed(self, seed=None):
        seed = seed or self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def _define_placeholders(self):
        # tf placeholders
        self.X = tf.placeholder(tf.int32, [None, self.config.max_length, 2])  # token idxs (BPE embedding + positional)
        self.M = tf.placeholder(tf.float32, [None, self.config.max_length])  # sequence mask
        # when target dim is not set, an array of [None] targets is passed as a placeholder
        self.Y = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.target_dim or 1]))  # classification targets
        self.do_dropout = tf.placeholder(tf.float32)  # 1 for do dropout and 0 to not do dropout
        if self.target_type == SEQUENCE_LABELING:
            self.Y = tf.placeholder(tf.int32, [None, self.config.max_length])  # classification targets
        else:
            self.Y = tf.placeholder(tf.float32, [None, self.target_dim])  # classification targets

    def generate_text(self, max_length=None, seed_text=''):
        """
        Performs a prediction on the Language modeling objective given some seed text. It uses a noisy greedy decoding.
        Temperature parameter for decoding is set in the config.

        :param max_length: The maximum length to decode to.
        :param seed_text: Defaults to the empty string. This will form the starting point to begin modelling
        :return: A string containing the generated text.
        """
        seed_text_tokens = self.encoder._encode([seed_text]).token_ids
        self._build_model(n_updates_total=0, target_dim=self.target_dim, train=False)
        string = [self.encoder['_start_']] + seed_text_tokens[0]
        eos = self.encoder['_classify_']
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for i in range(len(seed_text_tokens[0]), (max_length or self.config.max_length) - 1):
                model_input = self._array_format([string])
                class_idx = self.sess.run(self.lm_predict_op, {self.X: model_input.token_ids, self.M: model_input.mask})
                string.append(class_idx[i])
                if string[-1] == eos:
                    break
        return self.encoder.decode(string)

    def _load_base_model(self):
        """
        Load serialized base model parameters into tf Tensors
        """
        pretrained_params = find_trainable_variables('model', exclude='model/clf')
        self._initialize_session()
        self.sess.run(tf.global_variables_initializer())

        with open(SHAPES_PATH) as shapes_file:
            shapes = json.load(shapes_file)
            offsets = np.cumsum([np.prod(shape) for shape in shapes])
            init_params = [np.load(PARAM_PATH.format(n)) for n in range(10)]
            init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
            init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
            init_params[0] = init_params[0][:self.config.max_length]
            special_embed = (np.random.randn(len(self.encoder.special_tokens),
                                             self.config.n_embed) * self.config.weight_stddev).astype(np.float32)
            init_params[0] = np.concatenate([init_params[1], special_embed, init_params[0]], 0)
            del init_params[1]

            self.sess.run([p.assign(ip) for p, ip in zip(pretrained_params, init_params)])

    def __getstate__(self):
        """
        Leave serialization of all tf objects to tf
        """
        required_fields = [
            'label_encoder', 'target_dim', '_load_from_file', 'config', 'target_type',
        ]
        serialized_state = {
            k: v for k, v in self.__dict__.items()
            if k in required_fields
        }
        return serialized_state

    def save(self, path):
        """
        Saves the state of the model to disk in a location with name :param: path. The model is saved in several files
        and :param: path is used as a prefix.

        Save is performed in two steps:
            - Serialize tf graph to disk using tf.Saver
            - Serialize python model using pickle

        Note:
            Does not serialize state of Adam optimizer.
            Should not be used to save / restore a training model.
        """
        if path is None:
            return

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
        self._initialize_session()
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess, self._load_from_file)
        self._load_from_file = False
        self.is_trained = True

from finetune.base import BaseModel
from finetune.target_encoders import OneHotLabelEncoder
from finetune.network_modules import cosine_similarity

import os
import warnings
import logging
from collections import defaultdict

import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from finetune.network_modules import featurizer, language_model
from finetune.utils import (
    find_trainable_variables, assign_to_gpu, iter_data,
    soft_split, concat_or_stack, sample_with_temperature
)
from finetune.encoding import EncodedOutput
from finetune.imbalance import compute_class_weights, class_weight_tensor
from finetune.errors import FinetuneError

JL_BASE = os.path.join(os.path.dirname(__file__), "model", "Base_model.jl")

_LOGGER = logging.getLogger(__name__)

DROPOUT_ON = 1
DROPOUT_OFF = 0
SAVE_PREFIX = 'model'
MIN_UPDATES = 15


class SiameseComparison(BaseModel):
    """
    Compares two documents and produces a similarity score (between 0-1).

    :param config: A :py:class:`finetune.config.Settings` object or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def _text_to_ids(self, pairs, Y=None, max_length=None):
        """
        Format comparison examples as a list of IDs

        pairs: Array of text, shape [batch, 2]
        """
        return super()._text_to_ids(pairs[0], Y=Y, max_length=max_length), \
               super()._text_to_ids(pairs[1], Y=Y, max_length=max_length)

    def _construct_graph(self, n_updates_total, target_dim=None, train=True):
        gpu_grads = []
        self.summaries = []

        # store whether or not graph was previously compiled with dropout
        self.train = train
        self._define_placeholders(target_dim=target_dim)

        aggregator = defaultdict(list)
        train_loss_tower = 0
        gpus = self.config.visible_gpus
        n_splits = max(len(gpus), 1)

        # multi-GPU setup, using CPU as param server is most efficient unless system has direct GPU connections
        # single GPU, no need to use a different GPU as a parameter server
        params_device = 'cpu' if len(gpus) != 1 else gpus[0]

        # decide on setting for language model loss coefficient
        # if the language model loss does not contribute to overall loss,
        # remove the language model computation from the graph
        lm_loss_coef = self.config.lm_loss_coef
        if target_dim is None:
            lm_loss_coef = 1.0
        compile_lm = (train and lm_loss_coef > 0) or self.require_lm

        for i, (X1, M1, X2, M2, Y) in enumerate(soft_split(self.X1, self.M1, self.X2, self.M2, self.Y, n_splits=n_splits)):
            do_reuse = True if i > 0 else tf.AUTO_REUSE

            if gpus:
                device = tf.device(assign_to_gpu(gpus[i], params_device=params_device))
            else:
                device = tf.device('cpu')

            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)

            with device, scope:
                featurizer_state1 = featurizer(
                    X1,
                    config=self.config,
                    encoder=self.encoder,
                    dropout_placeholder=self.do_dropout,
                    train=train,
                    reuse=do_reuse
                )

                # TODO what does the reuse parameter do here?
                # Setting `reuse=1` here assuming it means the siamese networks share weights...
                featurizer_state2 = featurizer(
                    X2,
                    config=self.config,
                    encoder=self.encoder,
                    dropout_placeholder=self.do_dropout,
                    train=train,
                    reuse=1
                )

                if compile_lm:
                    # Language modelling is only performed on X1.
                    language_model_state = language_model(
                        X=X1,
                        M=M1,
                        config=self.config,
                        embed_weights=featurizer_state1['embed_weights'],
                        hidden=featurizer_state1['sequence_features'],
                        reuse=do_reuse
                    )

                    train_loss = lm_loss_coef * tf.reduce_mean(language_model_state['losses'])
                    aggregator['lm_losses'].append(language_model_state['losses'])
                    lm_logits = language_model_state["logits"]

                    lm_logit_mask = np.zeros([1, lm_logits.get_shape().as_list()[-1]], dtype=np.float32)
                    lm_logit_mask[:, self.encoder.vocab_size:] = -np.inf
                    lm_logits += lm_logit_mask

                    if "use_extra_toks" in self.config and not self.config.use_extra_toks:
                        lm_logit_mask[:, self.encoder.start] = -np.inf
                        lm_logit_mask[:, self.encoder.delimiter] = -np.inf
                        lm_logit_mask[:, self.encoder.clf_token] = -np.inf

                    aggregator["lm_model"].append(sample_with_temperature(lm_logits, self.config.lm_temp))
                else:
                    train_loss = 0

                aggregator['features1'].append(featurizer_state1['features'])
                aggregator['features2'].append(featurizer_state2['features'])

                if target_dim is not None:

                    weighted_tensor = None
                    if self.config.class_weights is not None:
                        weighted_tensor = class_weight_tensor(
                            class_weights=self.config.class_weights,
                            target_dim=target_dim,
                            label_encoder=self.label_encoder
                        )

                    with tf.variable_scope('model/target'):
                        target_model_config = {
                            'featurizer_state1': featurizer_state1,
                            'featurizer_state2': featurizer_state2,
                            'targets': Y,
                            'n_outputs': target_dim,
                            'train': train,
                            'reuse': do_reuse,
                            'max_length': self.config.max_length,
                            'class_weights': weighted_tensor
                        }
                        target_model_state = self._target_model(**target_model_config)
                    train_loss += (1 - lm_loss_coef) * tf.reduce_mean(target_model_state['losses'])
                    train_loss_tower += train_loss

                    aggregator['logits'].append(target_model_state['logits'])
                    aggregator['target_losses'].append(target_model_state['losses'])

                params = find_trainable_variables("model")
                grads = tf.gradients(train_loss, params)
                grads = list(zip(grads, params))
                gpu_grads.append(grads)

        with tf.device(params_device):
            self.features1 = tf.concat(aggregator['features1'], axis=0)
            self.features2 = tf.concat(aggregator['features2'], axis=0)

            if compile_lm:
                self.lm_predict_op = tf.concat(aggregator["lm_model"], 0)
                self.lm_losses = tf.concat(aggregator['lm_losses'], axis=0)
                self.lm_loss = tf.reduce_mean(self.lm_losses)
                self.summaries.append(tf.summary.scalar('LanguageModelLoss', self.lm_loss))

            if train:
                self._compile_train_op(
                    params=params,
                    grads=gpu_grads,
                    n_updates_total=n_updates_total
                )

            if target_dim is not None:
                self.logits = tf.concat(aggregator['logits'], axis=0)
                self.target_losses = concat_or_stack(aggregator['target_losses'])

                self.predict_op = self._predict_op(
                    self.logits, **target_model_state.get("predict_params", {})
                )
                self.predict_proba_op = self._predict_proba_op(
                    self.logits, **target_model_state.get("predict_params", {})
                )
                self.target_loss = tf.reduce_mean(self.target_losses)

                self.summaries.append(tf.summary.scalar('TargetModelLoss', self.target_loss))
                self.summaries.append(tf.summary.scalar('TotalLoss', train_loss_tower / n_splits))

            self.summaries = tf.summary.merge(self.summaries) if self.summaries else self.noop

    def _define_placeholders(self, target_dim=None):
        # tf placeholders
        self.X1 = tf.placeholder(tf.int32, [None, self.config.max_length, 2])  # token idxs (BPE embedding + positional)
        self.M1 = tf.placeholder(tf.float32, [None, self.config.max_length])  # sequence mask
        self.X2 = tf.placeholder(tf.int32, [None, self.config.max_length, 2])  # token idxs (BPE embedding + positional)
        self.M2 = tf.placeholder(tf.float32, [None, self.config.max_length])  # sequence mask
        # when target dim is not set, an array of [None] targets is passed as a placeholder

        self.do_dropout = tf.placeholder(tf.float32)  # 1 for do dropout and 0 to not do dropout
        self.Y = self._target_placeholder(target_dim=target_dim)

    def _target_placeholder(self, target_dim=None):
        return tf.placeholder(tf.float32, [None, target_dim or 1])

    def generate_text(self, seed_text='', max_length=None, use_extra_toks=True):
        """
        Performs a prediction on the Language modeling objective given some seed text. It uses a noisy greedy decoding.
        Temperature parameter for decoding is set in the config.

        :param max_length: The maximum length to decode to.
        :param seed_text: Defaults to the empty string. This will form the starting point to begin modelling
        :return: A string containing the generated text.
        """
        self.require_lm = True
        self.config.use_extra_toks = use_extra_toks
        encoded = self.encoder._encode([seed_text])
        if encoded == [] and not use_extra_toks:
            raise ValueError("If you are not using the extra tokens, you must provide some non-empty seed text")
        start = [self.encoder.start] if use_extra_toks else []
        encoded = EncodedOutput(token_ids=[start + encoded.token_ids[0]])
        self._build_model(n_updates_total=0, target_dim=self.target_dim, train=False)
        EOS = self.encoder.clf_token
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for i in range(len(encoded.token_ids), (max_length or self.config.max_length) - 2):
                arr_encoded = self._array_format(encoded)
                class_idx = self.sess.run(self.lm_predict_op, {self.X1: arr_encoded.token_ids, self.M1: arr_encoded.mask})
                encoded.token_ids[0].append(class_idx[i])
                if encoded.token_ids[0][-1] == EOS:
                    break

        del self.config["use_extra_toks"]
        return self.encoder.decode(encoded.token_ids[0])

    def _featurize(self, Xs, max_length=None):
        features = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.config.max_length
            for xmb, mmb in self._infer_prep_single(Xs, max_length=max_length):
                feature_batch = self.sess.run(self.features1, {
                    self.X1: xmb,
                    self.M1: mmb,
                    self.do_dropout: DROPOUT_OFF
                })
                features.append(feature_batch)
        return np.concatenate(features)

    def featurize(self, X, max_length):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :math:`finetune`.

        :param X: list or array of text to embed.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X, max_length)

    def _infer_prep_pairs(self, Xs, max_length=None):
        """
        Infers and prepares data for pairs of documents to be compared.

        :param Xs: list or array of lists of two texts to prepare.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :return: an iterator over the pairwise, prepared data.
        """
        max_length = max_length or self.config.max_length
        arr_encoded1, arr_encoded2 = self._text_to_ids(Xs, max_length=max_length)
        n_batch_train = self.config.batch_size * max(len(self.config.visible_gpus), 1)
        self._build_model(n_updates_total=0, target_dim=self.target_dim, train=False)
        yield from iter_data(arr_encoded1.token_ids, arr_encoded1.mask, arr_encoded2.token_ids, arr_encoded2.mask,
                             n_batch=n_batch_train,
                             verbose=self.config.verbose)

    def _infer_prep_single(self, Xs, max_length=None):
        """
        Infers and prepares data for a single document.

        :param Xs: list or array of text to prepare.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :return: an iterator over the prepared data.
        """
        max_length = max_length or self.config.max_length
        arr_encoded = super()._text_to_ids(Xs, max_length=max_length)
        n_batch_train = self.config.batch_size * max(len(self.config.visible_gpus), 1)
        self._build_model(n_updates_total=0, target_dim=self.target_dim, train=False)
        yield from iter_data(arr_encoded.token_ids, arr_encoded.mask, n_batch=n_batch_train,
                             verbose=self.config.verbose)

    def finetune(self, Xs, Y=None, batch_size=None):
        """
        :param Xs: list or array of list of two texts.
        :param Y: integer or string-valued class labels.
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        if Y is not None and len(Xs[0]) != len(Y):
            raise FinetuneError(
                "Mismatch between number of examples ({}) and number of targets ({}) provided.".format(
                    len(Xs[0]),
                    len(Y)
                )
            )
        arr_encoded = self._text_to_ids(Xs)
        return self._training_loop(
            arr_encoded,
            Y=Y,
            batch_size=batch_size,
        )

    def _training_loop(self, arr_encoded, Y=None, batch_size=None):
        # Retrieve encoded text 1 and 2
        arr_encoded1, arr_encoded2 = arr_encoded[0], arr_encoded[1]

        self.label_encoder = self._target_encoder()

        idxs = list(range(len(arr_encoded1.token_ids)))
        train_idxs, val_idxs = train_test_split(idxs, test_size=self.config.val_size)

        if Y is None:
            # only language model will be trained, mock fake target of right length
            train_Y = np.asarray([[]] * len(train_idxs))
            val_Y = np.asarray([[]] * len(val_idxs))
            target_dim = None
        else:
            Y = np.asarray(Y)
            train_Y = self.label_encoder.fit_transform(Y[train_idxs])
            val_Y = self.label_encoder.transform(Y[val_idxs])
            target_dim = self.label_encoder.target_dim

        batch_size = batch_size or self.config.batch_size
        n_batch_train = batch_size * max(len(self.config.visible_gpus), 1)
        n_examples = len(train_idxs)
        n_updates_total = (n_examples // n_batch_train) * self.config.n_epochs

        if (n_updates_total) <= MIN_UPDATES:
            warnings.warn(
                "Model will only receive {} weight updates.  This may not be sufficient to find a good minima."
                "Please consider lowering `config.batch_size` or providing more labeled training data to thet model."
            )

        train_dataset = (arr_encoded1.token_ids[train_idxs], arr_encoded1.mask[train_idxs],
                         arr_encoded2.token_ids[train_idxs], arr_encoded2.mask[train_idxs],
                         train_Y)
        val_dataset = (arr_encoded1.token_ids[val_idxs], arr_encoded2.mask[val_idxs],
                       arr_encoded1.token_ids[val_idxs], arr_encoded2.mask[val_idxs],
                       val_Y)

        self.config.class_weights = compute_class_weights(
            class_weights=self.config.class_weights,
            Y=Y
        )
        self._build_model(n_updates_total=n_updates_total, target_dim=target_dim)
        self.is_trained = True

        avg_train_loss = None
        avg_val_loss = None
        global_step = 0
        best_val_loss = float("inf")
        val_window = [float("inf")] * self.config.val_window_size

        for i in range(self.config.n_epochs):
            iterator = iter_data(
                *train_dataset,
                n_batch=n_batch_train,
                tqdm_desc="Epoch {}".format(i),
                verbose=self.config.verbose
            )
            for (xmb1, mmb1, xmb2, mmb2, ymb) in iterator:
                feed_dict = {
                    self.X1: xmb1,
                    self.M1: mmb1,
                    self.X2: xmb2,
                    self.M2: mmb2,
                }
                if target_dim:
                    feed_dict[self.Y] = ymb

                global_step += 1
                if global_step % self.config.val_interval == 0:
                    feed_dict[self.do_dropout] = DROPOUT_OFF

                    outputs = self._eval(self.summaries, feed_dict=feed_dict)
                    if self.train_writer is not None:
                        self.train_writer.add_summary(outputs.get(self.summaries), global_step)

                    sum_val_loss = 0
                    for xval1, mval1, xval2, mval2, yval in iter_data(
                            *val_dataset, n_batch=n_batch_train, verbose=self.config.verbose, tqdm_desc="Validation"):
                        feed_dict = {
                            self.X1: xval1,
                            self.M1: mval1,
                            self.X2: xval2,
                            self.M2: mval2,
                            self.do_dropout: DROPOUT_OFF
                        }
                        if target_dim:
                            feed_dict[self.Y] = yval

                        outputs = self._eval(self.target_loss, self.summaries, feed_dict=feed_dict)
                        if self.valid_writer is not None:
                            self.valid_writer.add_summary(outputs.get(self.summaries), global_step)

                        val_cost = outputs.get(self.target_loss, 0)
                        sum_val_loss += val_cost

                        if avg_val_loss is None:
                            avg_val_loss = val_cost
                        else:
                            avg_val_loss = (
                                    avg_val_loss * self.config.rolling_avg_decay
                                    + val_cost * (1 - self.config.rolling_avg_decay)
                            )
                    val_window.append(sum_val_loss)
                    val_window.pop(0)

                    if np.mean(val_window) <= best_val_loss:
                        best_val_loss = np.mean(val_window)
                        if self.config.autosave_path is not None:
                            self.save(self.config.autosave_path)

                    tqdm.tqdm.write("Train loss: {}\t Validation loss: {}".format(avg_train_loss, avg_val_loss))

                feed_dict[self.do_dropout] = DROPOUT_ON
                outputs = self._eval(self.target_loss, self.train_op, feed_dict=feed_dict)

                cost = outputs.get(self.target_loss, 0)
                if avg_train_loss is None:
                    avg_train_loss = cost
                else:
                    avg_train_loss = avg_train_loss * self.config.rolling_avg_decay + cost * (
                            1 - self.config.rolling_avg_decay)

        return self

    def _predict(self, Xs, max_length=None):
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.config.max_length
            for xmb1, mmb1, xmb2, mmb2 in self._infer_prep_pairs(Xs, max_length=max_length):
                output = self._eval(self.predict_op,
                    feed_dict={
                        self.X1: xmb1,
                        self.M1: mmb1,
                        self.X2: xmb2,
                        self.M2: mmb2,
                        self.do_dropout: DROPOUT_OFF
                    }
                )
                prediction = output.get(self.predict_op)
                formatted_predictions = self.label_encoder.inverse_transform(prediction)
                predictions.append(formatted_predictions)
        return np.concatenate(predictions).tolist()

    def predict(self, Xs, max_length=None):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of list of two texts to predict for.
        :param max_length: the number of tokens to be included in the document representation.
                           Providing more than `max_length` tokens as input will result in truncation.
        :returns: list of class labels.
        """
        return self._predict(Xs, max_length=max_length)

    def _predict_proba(self, Xs, max_length=None):
        """
        Produce raw numeric outputs for proba predictions
        """
        predictions = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            max_length = max_length or self.config.max_length
            for xmb1, mmb1, xmb2, mmb2 in self._infer_prep_pairs(Xs, max_length=max_length):
                output = self._eval(
                    self.predict_proba_op,
                    feed_dict={
                        self.X1: xmb1,
                        self.M1: mmb1,
                        self.X2: xmb2,
                        self.M2: mmb2,
                        self.do_dropout: DROPOUT_OFF
                    }
                )
                probas = output.get(self.predict_proba_op)
                predictions.extend(probas)
        return predictions

    def predict_proba(self, *args, **kwargs):
        """
        The base method for predicting from the model.
        """
        raw_probas = self._predict_proba(*args, **kwargs)
        classes = self.label_encoder.classes_

        formatted_predictions = []
        for probas in raw_probas:
            formatted_predictions.append(
                dict(zip(classes, probas))
            )
        return formatted_predictions

    def get_eval_fn(cls):
        return lambda labels, targets: np.mean(np.asarray(labels) == np.asarray(targets))

    def _target_encoder(self):
        return OneHotLabelEncoder()

    def _target_model(self, featurizer_state1, featurizer_state2, targets, n_outputs, train=False, reuse=None, **kwargs):
        return cosine_similarity(
            hidden1=featurizer_state1['features'],
            hidden2=featurizer_state2['features'],
            targets=targets,
            n_targets=n_outputs,
            dropout_placeholder=self.do_dropout,
            config=self.config,
            train=train,
            reuse=reuse,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        return tf.argmax(logits, -1)

    def _predict_proba_op(self, logits, **kwargs):
        return tf.nn.softmax(logits, -1)

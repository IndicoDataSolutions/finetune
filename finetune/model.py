from collections import defaultdict, namedtuple
import enum
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.train import Scaffold

from finetune.network_modules import featurizer, language_model
from finetune.utils import (
    find_trainable_variables, assign_to_gpu, soft_split, concat_or_stack, sample_with_temperature, average_grads
)
from finetune.optimizers import AdamWeightDecay, schedules
from finetune.imbalance import class_weight_tensor

DROPOUT_ON = 1
DROPOUT_OFF = 0


class PredictMode:
    FEATURIZE = "FEAT"
    NORMAL = "NORM"
    PROBAS = "PROBA"
    GENERATE_TEXT = "GEN_TEXT"


def get_model_fn(target_model_fn, predict_op, predict_proba_op, build_target_model, build_lm, encoder, target_dim,
                 label_encoder, saver):
    def get_train_op(*, variables, grads, n_updates_total, params):
        grads = average_grads(grads)

        if params.summarize_grads:
            tf.contrib.training.add_gradients_summaries(grads)

        grads = [grad for grad, param in grads]
        train_op = AdamWeightDecay(
            params=variables,
            grads=grads,
            lr=params.lr,
            schedule=partial(schedules[params.lr_schedule], warmup=params.lr_warmup),
            t_total=n_updates_total,
            l2=params.l2_reg,
            max_grad_norm=params.max_grad_norm,
            vector_l2=params.vector_l2,
            b1=params.b1,
            b2=params.b2,
            e=params.epsilon,
            pretrained_weights=saver.get_pretrained_weights(),
            deviation_regularization=params.regularize_deviation
        )
        return train_op

    def language_model_op(X, M, params, featurizer_state, do_reuse):
        language_model_state = language_model(
            X=X,
            M=M,
            config=params,
            embed_weights=featurizer_state['embed_weights'],
            hidden=featurizer_state['sequence_features'],
            reuse=do_reuse
        )

        lm_logits = language_model_state["logits"]

        lm_logit_mask = np.zeros([1, lm_logits.get_shape().as_list()[-1]], dtype=np.float32)
        lm_logit_mask[:, encoder.vocab_size:] = -np.inf

        if "use_extra_toks" in params and not params.use_extra_toks:
            lm_logit_mask[:, encoder.start] = -np.inf
            lm_logit_mask[:, encoder.delimiter] = -np.inf
            lm_logit_mask[:, encoder.clf_token] = -np.inf

        lm_logits += lm_logit_mask

        lm_predict_op = sample_with_temperature(lm_logits, params.lm_temp)
        return lm_predict_op, language_model_state

    def target_model_op(featurizer_state, Y, params, mode, do_reuse):
        weighted_tensor = None
        if params.class_weights is not None:
            weighted_tensor = class_weight_tensor(
                class_weights=params.class_weights,
                target_dim=target_dim,
                label_encoder=label_encoder
            )

        with tf.variable_scope('model/target'):
            target_model_config = {
                'featurizer_state': featurizer_state,
                'targets': Y,
                'n_outputs': target_dim,
                'train': mode == tf.estimator.ModeKeys.TRAIN,
                'reuse': do_reuse,
                'max_length': params.max_length,
                'class_weights': weighted_tensor
            }
            target_model_state = target_model_fn(**target_model_config)
        return target_model_state

    def _model_fn(features, labels, mode, params):
        if not build_target_model:
            lm_loss_coef = 1.
        else:
            lm_loss_coef = params.lm_loss_coef

        estimator_mode = mode
        train = estimator_mode == tf.estimator.ModeKeys.TRAIN
        aggregator = defaultdict(list)

        train_loss_tower = 0.
        lm_loss_tower = 0.
        target_loss_tower = 0.
        gpus = params.visible_gpus
        gpu_grads = []
        n_splits = max(len(gpus), 1)

        Xs = features["tokens"]
        Ms = features["mask"]

        for i, (X, M, Y) in enumerate(soft_split(Xs, Ms, labels, n_splits=n_splits)):
            do_reuse = True if i > 0 else tf.AUTO_REUSE

            if gpus:
                device = tf.device(assign_to_gpu(gpus[i], params_device=params.params_device))
            else:
                device = tf.device('cpu')

            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)

            with device, scope:
                train_loss = 0.0
                featurizer_state = featurizer(
                    X,
                    config=params,
                    encoder=encoder,
                    train=train,
                    reuse=do_reuse
                )
                aggregator['features'].append(featurizer_state['features'])

                if build_target_model:
                    target_model_state = target_model_op(featurizer_state=featurizer_state, Y=Y, params=params,
                                                         mode=mode,
                                                         do_reuse=do_reuse)
                    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                        train_loss += (1 - lm_loss_coef) * tf.reduce_mean(target_model_state['losses'])
                        target_loss_tower += tf.reduce_mean(target_model_state['losses'])
                        train_loss_tower += train_loss
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        aggregator['logits'].append(target_model_state['logits'])

                if build_lm:
                    lm_predict_op, language_model_state = language_model_op(X=X, M=M, params=params,
                                                                            featurizer_state=featurizer_state,
                                                                            do_reuse=do_reuse)
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        aggregator["lm_model"].append(lm_predict_op)

                    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                        train_loss += lm_loss_coef * tf.reduce_mean(language_model_state['losses'])
                        lm_loss_tower += language_model_state['losses']

                if mode == tf.estimator.ModeKeys.TRAIN:
                    variables = find_trainable_variables("model")
                    grads = tf.gradients(train_loss, variables)
                    grads = list(zip(grads, variables))
                    gpu_grads.append(grads)

        with tf.device(params.params_device):
            features = tf.concat(aggregator['features'], axis=0)
            if build_lm:
                lm_predict_op = tf.concat(aggregator["lm_model"], 0)
                tf.summary.scalar("LanguageModelLoss", lm_loss_tower)
            else:
                lm_predict_op = tf.constant(0)

            if mode == tf.estimator.ModeKeys.TRAIN:
                variables = find_trainable_variables("model")
                train_op = get_train_op(variables=variables, grads=gpu_grads, n_updates_total=1000, params=params)
                # TODO figure out what to do about this. We dont always know how many updates we will get now.

            if build_target_model:

                target_loss = tf.reduce_mean(target_loss_tower)
                tf.summary.scalar("TargetModelLoss", target_loss)
                if mode == tf.estimator.ModeKeys.PREDICT:
                    logits = tf.concat(aggregator['logits'], axis=0)


        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            total_loss = train_loss_tower / n_splits

            tf.summary.scalar("TotalLoss", total_loss)

        scaffold = Scaffold(init_fn=saver.get_scaffold_initializer())

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    PredictMode.NORMAL: predict_op(logits, **target_model_state.get("predict_params", {})),
                    PredictMode.PROBAS: predict_proba_op(logits, **target_model_state.get("predict_params", {})),
                    PredictMode.GENERATE_TEXT: lm_predict_op,
                    PredictMode.FEATURIZE: features
                },
                scaffold=scaffold
            )

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, scaffold=scaffold)

        assert mode == tf.estimator.ModeKeys.EVAL
        return tf.estimator.EstimatorSpec(mode=mode, loss=target_loss, scaffold=scaffold)

    return _model_fn

import logging

import numpy as np
import tensorflow as tf
from tensorflow.train import Scaffold
from tensorflow.contrib.opt.python.training.weight_decay_optimizers import AdamWOptimizer

from finetune.network_modules import featurizer, language_model
from finetune.utils import sample_with_temperature
from finetune.optimizers import schedules
from finetune.imbalance import class_weight_tensor

LOGGER = logging.getLogger('finetune')

class PredictMode:
    FEATURIZE = "FEAT"
    NORMAL = "NORM"
    PROBAS = "PROBA"
    GENERATE_TEXT = "GEN_TEXT"


def get_model_fn(target_model_fn, predict_op, predict_proba_op, build_target_model, build_lm, encoder, target_dim,
                 label_encoder, saver):
    def language_model_op(X, M, params, featurizer_state):
        language_model_state = language_model(
            X=X,
            M=M,
            config=params,
            embed_weights=featurizer_state['embed_weights'],
            hidden=featurizer_state['sequence_features'],
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

    def target_model_op(featurizer_state, Y, params, mode):
        weighted_tensor = None
        if params.class_weights is not None:
            weighted_tensor = class_weight_tensor(
                class_weights=params.class_weights,
                target_dim=target_dim,
                label_encoder=label_encoder
            )
        with tf.variable_scope('model/target'):
            target_model_state = target_model_fn(
                featurizer_state=featurizer_state,
                targets=Y,
                n_outputs=target_dim,
                train=mode == tf.estimator.ModeKeys.TRAIN,
                max_length=params.max_length,
                class_weights=weighted_tensor
            )
        return target_model_state

    def _model_fn(features, labels, mode, params):
        if "labels" in features:
            assert labels is None, "For some reason distributed tensorflow doesnt let us use labels argument"
            labels = features["labels"]

        if not build_target_model:
            lm_loss_coef = 1.
        else:
            lm_loss_coef = params.lm_loss_coef

        estimator_mode = mode
        train = estimator_mode == tf.estimator.ModeKeys.TRAIN
        X = features["tokens"]
        M = features["mask"]
        Y = labels
        pred_op = None

        with tf.variable_scope(tf.get_variable_scope()):
            train_loss = 0.0
            featurizer_state = featurizer(X, config=params, encoder=encoder, train=train)
            predictions = {PredictMode.FEATURIZE: featurizer_state["features"]}

            if build_target_model:
                target_model_state = target_model_op(featurizer_state=featurizer_state, Y=Y, params=params, mode=mode)
                if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL) and Y is not None:
                    target_loss = tf.reduce_mean(target_model_state["losses"])
                    train_loss += (1 - lm_loss_coef) * target_loss
                    tf.summary.scalar("TargetModelLoss", target_loss)
                if mode == tf.estimator.ModeKeys.PREDICT or tf.estimator.ModeKeys.EVAL:
                    logits = target_model_state["logits"]
                    predict_params = target_model_state.get("predict_params", {})
                    if "_threshold" in params:
                        predict_params["threshold"] = params._threshold
                    pred_op = predict_op(logits, **predict_params)
                    if type(pred_op) == tuple:
                        pred_op, pred_proba_op = pred_op
                    else:
                        pred_proba_op = predict_proba_op(logits, **predict_params)

                    predictions[PredictMode.NORMAL] = pred_op
                    predictions[PredictMode.PROBAS] = pred_proba_op

            if build_lm:
                lm_predict_op, language_model_state = language_model_op(X=X, M=M, params=params,
                                                                        featurizer_state=featurizer_state)
                if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                    lm_loss = tf.reduce_mean(language_model_state["losses"])
                    train_loss += lm_loss_coef * lm_loss
                    tf.summary.scalar("LanguageModelLoss", lm_loss)
                if mode == tf.estimator.ModeKeys.PREDICT:
                    predictions[PredictMode.GENERATE_TEXT] = lm_predict_op

        if mode == tf.estimator.ModeKeys.TRAIN:
            total_num_steps = params.n_epochs * params.dataset_size//params.batch_size
            lr_decay = lambda lr, global_step: lr * schedules[params.lr_schedule](tf.to_float(global_step) / total_num_steps)
            
            optimizer = lambda lr: AdamWOptimizer(
                learning_rate=lr,
                beta1=params.b1,
                beta2=params.b2,
                epsilon=params.epsilon,
                weight_decay=params.l2_reg * lr
            )

            summaries = tf.contrib.layers.OPTIMIZER_SUMMARIES if params.summarize_grads else None
            train_op = tf.contrib.layers.optimize_loss(
                loss=train_loss,
                global_step=tf.train.get_or_create_global_step(),
                learning_rate=params.lr,
                optimizer=optimizer,
                clip_gradients=float(params.max_grad_norm),
                learning_rate_decay_fn=lr_decay,
                increment_global_step=True,
                summaries=summaries
            )

        init_op = saver.get_scaffold_init_op()
        scaffold = Scaffold(init_op=init_op)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold=scaffold
            )

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=train_loss, train_op=train_op, scaffold=scaffold)

        assert mode == tf.estimator.ModeKeys.EVAL, "The mode is actually {}".format(mode)
        if params.eval_acc and pred_op is not None:
            LOGGER.info("Adding evaluation metrics, Accuracy")
            labels_dense = tf.argmax(labels)
            metrics = {
                "Accuracy":  tf.metrics.accuracy(pred_op, labels_dense)
            }
        else:
            metrics = None

        return tf.estimator.EstimatorSpec(mode=mode, loss=train_loss, scaffold=scaffold, eval_metric_ops=metrics)

    return _model_fn

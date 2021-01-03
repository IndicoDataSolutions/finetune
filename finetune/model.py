import logging
import functools

import numpy as np
import tensorflow as tf


from finetune.nn.target_blocks import language_model, masked_language_model
from finetune.util.text_generation import sample_with_temperature
from finetune.util.optimize_loss import optimize_loss

from finetune.util.imbalance import class_weight_tensor
from finetune.errors import FinetuneError
from finetune.base_models import GPTModel, GPTModelSmall

LOGGER = logging.getLogger("finetune")

class PredictMode:
    FEATURIZE = "FEAT"
    NORMAL = "NORM"
    PROBAS = "PROBA"
    GENERATE_TEXT = "GEN_TEXT"
    LM_PERPLEXITY = "PERPLEXITY"
    ATTENTION = "ATTENTION"
    SEQUENCE = "SEQUENCE"
    SEQUENCE_PROBAS = "SEQUENCE_PROBA"
    ASSOCIATION = "ASSOCIATION"
    ASSOCIATION_PROBAS = "ASSOCIATION_PROBA"
    EXPLAIN = "EXPLAIN"

def fp16_variable_getter(getter, name, shape=None, dtype=None,
                         initializer=None, regularizer=None,
                         trainable=True,
                         *args, **kwargs):
    dtype = tf.float16 if dtype in [tf.float16, tf.float32] else dtype
    return getter(name, shape, dtype=dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)


def language_model_op(X, params, featurizer_state, mode, encoder):
    language_model_state = language_model(
        X=X,
        sequence_lengths=featurizer_state["lengths"],
        config=params,
        embed_weights=featurizer_state['embed_weights'],
        hidden=featurizer_state['sequence_features'],
        train=mode == tf.estimator.ModeKeys.TRAIN,
    )
    lm_logits = language_model_state["logits"]

    if lm_logits is not None:

        lm_logit_mask = np.zeros(
            [1, lm_logits.get_shape().as_list()[-1]], dtype=np.float32
        )
        lm_logit_mask[:, encoder.vocab_size :] = -np.inf

        if "use_extra_toks" in params and not params.use_extra_toks:
            lm_logit_mask[:, encoder.start_token] = -np.inf
            lm_logit_mask[:, encoder.delimiter_token] = -np.inf
            lm_logit_mask[:, encoder.end_token] = -np.inf

            lm_logits += lm_logit_mask
        lm_predict_op = sample_with_temperature(lm_logits, params.lm_temp)
    else:
        lm_predict_op = tf.no_op()
    return lm_predict_op, language_model_state


def masked_language_model_op(X, mlm_weights, mlm_ids, mlm_positions, params, featurizer_state, mode):
    return masked_language_model(
        X=X,
        mlm_weights=mlm_weights,
        mlm_ids=mlm_ids,
        mlm_positions=mlm_positions,
        config=params,
        embed_weights=featurizer_state["embed_weights"],
        hidden=featurizer_state["sequence_features"],
        train=(mode == tf.estimator.ModeKeys.TRAIN)
    )
    return language_model_state


def fp16_variable_getter(
        getter, name, shape=None, dtype=None,
        initializer=None, regularizer=None,
        trainable=True,
        *args, **kwargs
):
    return getter(
        name,
        shape,
        dtype=tf.float16 if dtype in [tf.float16, tf.float32] else dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        *args,
        **kwargs
    )


def get_variable_getter(estimator_mode, features, fp16_predict):
    if estimator_mode == tf.estimator.ModeKeys.PREDICT and fp16_predict:
        custom_getter = fp16_variable_getter
        features = tf.nest.map_structure(
            lambda feat: tf.cast(feat, tf.float16) if feat.dtype == tf.float32 else feat,
            features
        )
    else:
        custom_getter = None
    return custom_getter, features

def get_model_fn(
    target_model_fn,
    pre_target_model_hook,
    predict_op,
    predict_proba_op,
    build_target_model,
    lm_type,
    encoder,
    target_dim,
    label_encoder,
    build_explain,
    n_replicas,
    fp16_predict,
):
    def target_model_op(featurizer_state, Y, params, mode, **kwargs):
        weighted_tensor = None
        if params.class_weights is not None:
            weighted_tensor = class_weight_tensor(
                class_weights=params.class_weights,
                target_dim=target_dim,
                label_encoder=label_encoder,
            )
        with tf.compat.v1.variable_scope("model/target"):
            pre_target_model_hook(featurizer_state)
            target_model_state = target_model_fn(
                config=params,
                featurizer_state=featurizer_state,
                targets=Y,
                n_outputs=target_dim,
                train=(mode == tf.estimator.ModeKeys.TRAIN),
                max_length=params.max_length,
                class_weights=weighted_tensor,
                label_encoder=label_encoder,
                **kwargs
            )
            
        return target_model_state

    def _model_fn(features, labels, mode, params):
        var_getter, features = get_variable_getter(mode, features, fp16_predict)
        if not build_target_model:
            lm_loss_coef = 1.0
        else:
            lm_loss_coef = params.lm_loss_coef

        estimator_mode = mode
        train = estimator_mode == tf.estimator.ModeKeys.TRAIN
        X = features["tokens"]
        context = features.get("context", None)
        Y = labels
        pred_op = None

        if estimator_mode == tf.estimator.ModeKeys.PREDICT:
            total_num_steps = None
        else:
            total_num_steps = params.n_epochs * params.dataset_size // (params.batch_size * n_replicas)
            
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), custom_getter=var_getter):
            train_loss = 0.0
            featurizer_state = params.base_model.get_featurizer(
                X,
                encoder=encoder,
                config=params,
                train=train,
                explain=build_explain,
                context=context,
                total_num_steps=total_num_steps,
                lengths=features.get("length"),
            )
            predictions = {
                PredictMode.FEATURIZE: featurizer_state["features"], 
                PredictMode.SEQUENCE: featurizer_state["sequence_features"]
            }

            if params.base_model in [GPTModel, GPTModelSmall]:
                predictions[PredictMode.ATTENTION] = featurizer_state[
                    "attention_weights"
                ]

            if build_target_model:
                target_model_state = target_model_op(
                    featurizer_state=featurizer_state,
                    Y=Y,
                    params=params,
                    mode=mode,
                )
                if (
                    mode == tf.estimator.ModeKeys.TRAIN
                    or mode == tf.estimator.ModeKeys.EVAL
                ) and Y is not None:
                    target_loss = tf.reduce_mean(input_tensor=target_model_state["losses"])
                    train_loss += (1 - lm_loss_coef) * target_loss
                    tf.compat.v1.summary.scalar("TargetModelLoss", target_loss)
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

                    if type(pred_op) == dict:
                        predictions.update(pred_op)
                        predictions.update(pred_proba_op)
                    else:
                        predictions[PredictMode.NORMAL] = pred_op
                        predictions[PredictMode.PROBAS] = pred_proba_op

                    if build_explain:
                        predictions[PredictMode.EXPLAIN] = target_model_state[
                            "explanation"
                        ]

            if lm_type is not None:
                if lm_type.lower() == 'lm':
                    lm_predict_op, language_model_state = language_model_op(
                        X=X, params=params, featurizer_state=featurizer_state, mode=mode, encoder=encoder
                    )
                elif lm_type.lower() == 'mlm':
                    if "mlm_weights" not in features:
                        raise FinetuneError(
                            "MLM pretraining must be performed through MaskedLanguageModel model type,"
                            " please either provide targets or switch to the MaskedLanagugeModel."
                        )
                    language_model_state = masked_language_model_op(
                        X=X, 
                        mlm_weights=features['mlm_weights'],
                        mlm_ids=features['mlm_ids'],
                        mlm_positions=features['mlm_positions'],
                        params=params, 
                        featurizer_state=featurizer_state,
                        mode=mode
                    )
                    # No support for any form of text generation for MLM for now
                    lm_predict_op = None
                else: 
                    raise FinetuneError("Unsupport `lm_type` option: {}".format(lm_type))

                if (
                    mode == tf.estimator.ModeKeys.TRAIN
                    or mode == tf.estimator.ModeKeys.EVAL
                ):
                    lm_loss = tf.reduce_mean(input_tensor=language_model_state["losses"])
                    train_loss += lm_loss_coef * lm_loss
                    tf.compat.v1.summary.scalar("LanguageModelLoss", lm_loss)
                if mode == tf.estimator.ModeKeys.PREDICT:
                    if lm_predict_op is not None:
                        predictions[PredictMode.GENERATE_TEXT] = lm_predict_op
                    predictions[PredictMode.LM_PERPLEXITY] = language_model_state[
                        "perplexity"
                    ]

        if mode == tf.estimator.ModeKeys.TRAIN:
            total_num_steps = params.n_epochs * params.dataset_size // (params.batch_size * n_replicas)

            train_op = optimize_loss(
                loss=train_loss,
                learning_rate=params.lr,
                optimizer_name=params.optimizer,
                clip_gradients=float(params.max_grad_norm),
                lr_schedule=params.lr_schedule,
                lr_warmup=params.lr_warmup,
                total_num_steps=total_num_steps,
                summarize_grads=params.summarize_grads,
                scale_loss=params.scale_loss,
                b1=params.b1,
                b2=params.b2,
                epsilon=params.epsilon,
                l2_reg=params.l2_reg,
                vector_l2=params.vector_l2,
                accumulate_steps=params.accum_steps,
            )

        if mode == tf.estimator.ModeKeys.PREDICT:
            for k, v in predictions.items():
                if v.dtype == tf.float16:
                    predictions[k] = tf.cast(v, tf.float32)
                
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=train_loss, train_op=train_op
            )

        assert mode == tf.estimator.ModeKeys.EVAL, "The mode is actually {}".format(
            mode
        )
        if params.eval_acc and pred_op is not None:
            LOGGER.info("Adding evaluation metrics, Accuracy")
            labels_dense = tf.argmax(input=labels, axis=-1)
            metrics = {"Accuracy": tf.compat.v1.metrics.accuracy(tf.argmax(input=pred_op, axis=-1), labels_dense)}
        else:
            metrics = None

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=train_loss, eval_metric_ops=metrics
        )

    return _model_fn

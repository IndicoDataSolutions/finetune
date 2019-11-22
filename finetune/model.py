import logging
import functools

import numpy as np
import tensorflow as tf


from finetune.nn.target_blocks import language_model, masked_language_model
from finetune.util.text_generation import sample_with_temperature
from finetune.optimizers.zero_grad import dont_optimize_zeros
from finetune.optimizers.gradient_accumulation import get_grad_accumulation_optimizer
from finetune.optimizers.learning_rate_schedules import schedules
from finetune.optimizers.adamax import AdamaxWOptimizer
from finetune.optimizers.adamw import AdamWOptimizer
from finetune.util.imbalance import class_weight_tensor
from finetune.errors import FinetuneError
from finetune.base_models import GPTModel, GPTModelSmall
from finetune.optimizers.adafactor import AdafactorWOptimizer, AdafactorOptimizer
from finetune.nn.auxiliary import embed_context, embed_position

LOGGER = logging.getLogger("finetune")

OPTIMIZERS = {
    "AdamW": AdamWOptimizer,
    "AdamaxW": AdamaxWOptimizer,
    "AdafactorW": AdafactorWOptimizer,
    "Adafactor": AdafactorOptimizer
}

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
):
    def target_model_op(featurizer_state, Y, params, mode, **kwargs):
        weighted_tensor = None
        if params.class_weights is not None:
            weighted_tensor = class_weight_tensor(
                class_weights=params.class_weights,
                target_dim=target_dim,
                label_encoder=label_encoder,
            )
        with tf.variable_scope("model/target"):
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
        if not build_target_model:
            lm_loss_coef = 1.0
        else:
            lm_loss_coef = params.lm_loss_coef

        estimator_mode = mode
        train = estimator_mode == tf.estimator.ModeKeys.TRAIN
        X = features["tokens"]
        context = features.get("context", None)
        task_id = features.get("task_id", None)
        Y = labels
        pred_op = None

        with tf.variable_scope(tf.get_variable_scope()):
            train_loss = 0.0
            featurizer_state = params.base_model.get_featurizer(
                X,
                encoder=encoder,
                config=params,
                train=train,
                explain=build_explain,
            )
            if context is not None:
                print("Embedding position")
                embed_position(context, featurizer_state, params, train)
            
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
                    task_id=task_id,
                )
                if (
                    mode == tf.estimator.ModeKeys.TRAIN
                    or mode == tf.estimator.ModeKeys.EVAL
                ) and Y is not None:
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
                    lm_loss = tf.reduce_mean(language_model_state["losses"])
                    train_loss += lm_loss_coef * lm_loss
                    tf.summary.scalar("LanguageModelLoss", lm_loss)
                if mode == tf.estimator.ModeKeys.PREDICT:
                    if lm_predict_op is not None:
                        predictions[PredictMode.GENERATE_TEXT] = lm_predict_op
                    predictions[PredictMode.LM_PERPLEXITY] = language_model_state[
                        "perplexity"
                    ]

        if mode == tf.estimator.ModeKeys.TRAIN:
            total_num_steps = params.n_epochs * params.dataset_size // (params.batch_size * n_replicas)
            lr_decay = lambda lr, global_step: tf.maximum(
                0.0,
                lr
                * schedules[params.lr_schedule](
                    tf.to_float(global_step) / total_num_steps, warmup=params.lr_warmup
                ),
            )

            def optimizer(lr):
                Optimizer = OPTIMIZERS.get(params.optimizer, None)
                if Optimizer is None:
                    raise FinetuneError(
                        "Optimizer must be in {}, not {}".format(
                            list(OPTIMIZERS.keys()), params.optimizer
                        )
                    )

                if params.accum_steps > 1:
                    Optimizer = get_grad_accumulation_optimizer(
                        Optimizer, params.accum_steps
                    )

                opt = Optimizer(
                    learning_rate=lr,
                    beta1=params.b1,
                    beta2=params.b2,
                    epsilon=params.epsilon,
                    weight_decay=params.l2_reg * lr,
                )
                decay_var_list = [
                    v
                    for v in tf.global_variables()
                    if len(v.get_shape()) > 1 or params.vector_l2 and "OptimizeLoss" not in v.name
                ]

                opt.apply_gradients = functools.partial(
                    opt.apply_gradients, decay_var_list=decay_var_list
                )

                decay_var_list = [v for v in tf.global_variables() if len(v.get_shape()) > 1 or params.vector_l2]
                opt.apply_gradients = functools.partial(opt.apply_gradients, decay_var_list=decay_var_list)

                if params.scale_loss:
                    opt = tf.train.experimental.MixedPrecisionLossScaleOptimizer(opt, "dynamic")

                return opt

            summaries = (
                tf.contrib.layers.OPTIMIZER_SUMMARIES
                if params.summarize_grads
                else None
            )

            train_op = tf.contrib.layers.optimize_loss(
                loss=train_loss,
                global_step=tf.train.get_or_create_global_step(),
                learning_rate=tf.constant(params.lr),
                optimizer=optimizer,
                clip_gradients=float(params.max_grad_norm),
                learning_rate_decay_fn=lr_decay,
                increment_global_step=True,
                summaries=summaries,
                colocate_gradients_with_ops=True,
#                variables=trained_variables,
            )

        if mode == tf.estimator.ModeKeys.PREDICT:
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
            labels_dense = tf.argmax(labels, -1)
            metrics = {"Accuracy": tf.metrics.accuracy(tf.argmax(pred_op, -1), labels_dense)}
        else:
            metrics = None

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=train_loss, eval_metric_ops=metrics
        )

    return _model_fn

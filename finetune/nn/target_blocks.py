from finetune.nn.crf import crf_log_likelihood, sequence_decode
import tensorflow as tf
from finetune.nn.nn_utils import ExtraScope



class Perceptron(tf.keras.layers.Layer):
    def __init__(self, n_targets, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.n_targets), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.n_targets), initializer="zeros", trainable=True)
        super().build(input_shape)


    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def compute_loss(self, inputs, targets):
        raise NotImplementedError("Perceptron does not support compute_loss")


def _apply_class_weight(losses, targets, class_weights=None, norm_grads=True):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = tf.reduce_sum(
            input_tensor=class_weights * tf.cast(targets, dtype=tf.float32), axis=1
        )
        if norm_grads:
            weights *= tf.math.divide_no_nan(
                tf.cast(
                    tf.reduce_prod(input_tensor=tf.shape(input=weights)),
                    dtype=tf.float32,
                ),
                tf.reduce_sum(input_tensor=weights),
            )
        losses *= tf.expand_dims(weights, 1)
    return losses


def _apply_multilabel_class_weight(
    losses, targets, class_weights=None, norm_grads=True
):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = (
            # contribution of positive class
            class_weights * tf.cast(targets, dtype=tf.float32)
            +
            # contribution of negative class
            tf.ones_like(class_weights) * (1 - tf.cast(targets, dtype=tf.float32))
        )
        if norm_grads:
            weights *= tf.math.divide_no_nan(
                tf.cast(
                    tf.reduce_prod(input_tensor=tf.shape(input=weights)),
                    dtype=tf.float32,
                ),
                tf.reduce_sum(input_tensor=weights),
            )
        losses *= weights
    return losses

class MultiClassifier(tf.keras.layers.Layer):
    def __init__(self, n_targets, n_inputs, dropout_rate, renorm_after_class_weights, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.perceptron = Perceptron(n_targets, n_inputs)
        self.renorm_after_class_weights = renorm_after_class_weights
        self.threshold = threshold
    def call(self, inputs, training=False):
        inputs = self.dropout(inputs, training=training)
        logits = self.perceptron(inputs)
        return {
            "logits": logits,
            "probas": tf.nn.sigmoid(logits),
            "preds": tf.cast(tf.nn.sigmoid(logits) > self.threshold, tf.int32)
        }
                
    def compute_loss(self, layer_output, targets, class_weights):
        clf_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=layer_output["logits"], labels=tf.stop_gradient(targets)
            )
        clf_losses = _apply_multilabel_class_weight(
            clf_losses,
            targets,
            class_weights,
            norm_grads=self.renorm_after_class_weights,
        )
        return clf_losses
    
class Classifier(tf.keras.layers.Layer):
    def __init__(self, n_targets, n_inputs, dropout_rate, renorm_after_class_weights, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.perceptron = Perceptron(n_targets, n_inputs)
        self.renorm_after_class_weights = renorm_after_class_weights

    def call(self, inputs, training=False):
        inputs = self.dropout(inputs, training=training)
        logits = self.perceptron(inputs)
        return {
            "logits": logits,
            "probas": tf.nn.softmax(logits, -1),
            "preds": tf.argmax(logits, -1)
        }
    
    def compute_loss(self, layer_output, targets, class_weights):
        clf_losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=layer_output["logits"], labels=tf.stop_gradient(targets)
            )
        clf_losses = _apply_class_weight(
            clf_losses,
            targets,
            class_weights,
            norm_grads=self.renorm_after_class_weights,
        )
        return clf_losses


# TODO; historically custom_gradient had very poor performance compared to using a defun - double check this isn't still the case.
@tf.custom_gradient
def class_reweighted_grad(
    logits, class_weights, norm_grads_multiplier
):
    def custom_grad_fn(g):
        new_g = g * class_weights
        # This gets really badly autographed so we just need to use a float to cover these cases for now.
        ratio = tf.math.divide_no_nan(tf.norm(g), tf.norm(new_g)) * norm_grads_multiplier + (1 - norm_grads_multiplier)
        return [new_g * ratio, None, None]

    return tf.identity(logits), custom_grad_fn

class SequenceLabeler(tf.keras.layers.Layer):
    def __init__(self, n_targets, dropout_rate, use_crf, renorm_after_class_weights, name="sequence-labeler", **kwargs):
        super().__init__(**kwargs, name=name)
        self.n_targets = n_targets
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # This extra scope is an unfortunate holdover from GPT when we had an extra bidirectional attention block.
        self.dense = ExtraScope(tf.keras.layers.Dense(n_targets), "seq_lab_attn")
        self.use_crf = use_crf
        self.renorm_after_class_weights = renorm_after_class_weights

    def build(self, input_shape):
        if self.use_crf:
            self.transition_params = self.add_weight(
                shape=(self.n_targets, self.n_targets),
                # TODO: need to check what we have previously called this to make mapping easier
                # And likely move it into target blocks to get the scopes right.
                name="transition_matrix",
                initializer="orthogonal",
                trainable=True,
                dtype=tf.float32
            )
        else:
            self.transition_params = None
        super().build(input_shape)

    def call(self, inputs, training=False):
        logits = self.dense(inputs["sequence_features"])
        logits = tf.cast(logits, tf.float32)
        # CRF already outputs the format we need including probs, logits preds etc.
        preds = None
        probs = None
        if not training:
            preds, probs = sequence_decode(logits, self.transition_params, self.use_crf)
        return {
            "preds": preds,
            "probas": probs,
            "logits": logits,
            "length": inputs["length"],
            "transition_params": self.transition_params
        }

    def compute_loss(self, layer_output, targets, class_weights):
        # For some reason, all finetune targets are floats. I think we get more type flexibility 
        # now so we should look at switching this to int when helpful.
        print("output", layer_output["logits"].shape, layer_output["length"].shape, "targets", targets.shape)
        logits = layer_output["logits"]
        targets = tf.cast(targets, dtype=tf.int32)
        if class_weights is not None:
            class_weights = tf.reshape(class_weights, [1, 1, -1])
            one_hot_class_weights = class_weights * tf.one_hot(
                targets, depth=self.n_targets
            )
            per_token_weights = tf.reduce_sum(
                input_tensor=one_hot_class_weights, axis=-1, keepdims=True
            )
            logits = class_reweighted_grad(
                # You cannot use keyword arguments here. But the error message is horribly written.
                logits,
                per_token_weights,
                1.0 if self.renorm_after_class_weights else 0.0
            )

        if self.use_crf:
            tf.print("targets", targets)
            tf.print("logits", logits)
            return tf.reduce_mean(
                -crf_log_likelihood(
                    logits,
                    targets,
                    layer_output["length"],
                    layer_output["transition_params"]
                )
            )
        weights = tf.math.divide_no_nan(
            tf.sequence_mask(
                layer_output["length"],
                maxlen=tf.shape(input=targets)[1],
                dtype=tf.float32,
            ),
            tf.expand_dims(tf.cast(layer_output["length"], tf.float32), -1),
        )

        return tf.compat.v1.losses.sparse_softmax_cross_entropy(
            targets, logits, weights=weights
        )
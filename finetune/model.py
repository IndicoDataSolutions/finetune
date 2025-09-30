import logging
import time

import tensorflow as tf

from finetune.config import Settings
from finetune.encoding.input_encoder import BaseEncoder
from finetune.encoding.target_encoders import BaseEncoder as BaseTargetEncoder
from finetune.nn.nn_utils import ExtraScope
from finetune.util.imbalance import class_weight_tensor

LOGGER = logging.getLogger("finetune")


def get_keras_model(
    target_block: tf.keras.layers.Layer,
    encoder: BaseEncoder,
    target_dim: int,
    label_encoder: BaseTargetEncoder,
    config: Settings,
    train_input_signature: tuple[dict[str, tf.TensorSpec], tf.TensorSpec],
    predict_input_signature: dict[str, tf.TensorSpec],
    use_xla: bool,
    **model_kwargs
):
    class FinetuneModel(tf.keras.Model):
        def __init__(self, *args, name="model", **kwargs):
            super().__init__(*args, name=name, **kwargs)
            self.featurizer = config.base_model.get_featurizer(
                encoder=encoder, config=config, name="featurizer"
            )
            if target_block is not None:
                self.target_block = ExtraScope(target_block, "target")
            else:
                self.target_block = None

        def call(self, data, **kwargs):
            LOGGER.debug("Tracing Model Function")
            # Data contains:
            # * tokens
            # * length
            # * context (optional - only for models with context)
            # * row_gather (optional - only for the Table Model)
            # * col_gather (optional - only for the Table Model)

            data = dict(data)  # Keras doesn't like it if we modify the dict in place
            length = data.pop("length")
            data["sequence_lengths"] = length
            data["context"] = data.get("context", None)
            features: dict[str, tf.Tensor] = self.featurizer(**data, **kwargs)
            # Unpack to include things like lengths
            if self.target_block is not None:
                target_output: dict[str, tf.Tensor] = self.target_block(
                    {**features, "length": length}
                )
            else:
                target_output = {}
            output = {
                **features,
                **target_output,
            }
            # Certain keras calls assert that there is no None output.
            # We will just need to assume downstream that any missing values were None
            LOGGER.debug("Model function completed")
            return {k: v for k, v in output.items() if v is not None}

        def build(self, input_shape):
            pass

        def compute_loss(self, y, y_pred):
            weighted_tensor = None
            if config.class_weights is not None:
                weighted_tensor = class_weight_tensor(
                    class_weights=config.class_weights,
                    target_dim=target_dim,
                    label_encoder=label_encoder,
                )
            return self.target_block.compute_loss(
                layer_output=y_pred, targets=y, class_weights=weighted_tensor
            )

        @tf.function(
            input_signature=[predict_input_signature],
            autograph=False,
            jit_compile=use_xla,
        )
        def finetune_predict(self, data):
            return self.call(data, training=False)

        def fit(self, *args, **kwargs):
            # Build this explicitly outside the tf.function so that we don't need to re-trace.
            self.optimizer.build(self.trainable_variables)
            return super().fit(*args, **kwargs)

        @tf.function(jit_compile=True)
        def apply_gradients(self, grads_and_vars):
            return self.optimizer.apply_gradients(grads_and_vars)

        @tf.function(input_signature=[train_input_signature], autograph=False)
        def train_step(self, data):
            tic = time.time()
            LOGGER.debug("Tracing train step")
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                compute_loss_tic = time.time()
                loss = self.compute_loss(y=y, y_pred=y_pred)
                LOGGER.debug(
                    "Compute loss trace completed in %s seconds",
                    time.time() - compute_loss_tic,
                )
                # This is automatically stubbed out for default optimizers without scaling.
                loss = self.optimizer.scale_loss(loss)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients_tic = time.time()
            gradients = tape.gradient(loss, trainable_vars)
            LOGGER.debug(
                "Gradient trace completed in %s seconds", time.time() - gradients_tic
            )
            apply_gradients_tic = time.time()
            # Update weights
            self.apply_gradients(zip(gradients, trainable_vars))
            LOGGER.debug(
                "Apply gradients trace completed in %s seconds",
                time.time() - apply_gradients_tic,
            )
            LOGGER.debug("Train step trace completed in %s seconds", time.time() - tic)
            return {"loss": loss}

    return FinetuneModel(**model_kwargs)

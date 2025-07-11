import logging

import tensorflow as tf

from finetune.util.imbalance import class_weight_tensor
from finetune.encoding.input_encoder import BaseEncoder
from finetune.encoding.target_encoders import BaseEncoder as BaseTargetEncoder
from finetune.config import Settings
LOGGER = logging.getLogger("finetune")


def get_keras_model(
    target_block: tf.keras.layers.Layer,
    encoder: BaseEncoder,
    target_dim: int,
    label_encoder: BaseTargetEncoder,
    config: Settings,
    **kwargs
):
    
    
    class FinetuneModel(tf.keras.Model):
        def __init__(self, *args, name="model", **kwargs):
            super().__init__(*args, name=name, **kwargs)
            self.featurizer = config.base_model.get_featurizer(encoder=encoder, config=config, name="featurizer")
            self.target_block = target_block
            
        def call(self, data, **kwargs):
            features: dict[str, tf.Tensor] = self.featurizer(
                tokens=data["tokens"],
                context=data.get("context", None),
                sequence_lengths=data["length"],
                **kwargs
            )
            # Unpack to include things like lengths
            target_output: dict[str, tf.Tensor] = self.target_block({**features, **data})
            return {
                **features,
                **target_output,
            }
        
        def compute_loss(self, y, y_pred):
            weighted_tensor = None
            if config.class_weights is not None:
                weighted_tensor = class_weight_tensor(
                    class_weights=config.class_weights,
                    target_dim=target_dim,
                    label_encoder=label_encoder,
                )
            return self.target_block.compute_loss(layer_output=y_pred, targets=y, class_weights=weighted_tensor)
        
        def train_step(self, data):
            x, y = data
            tf.print("Y", y)

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(y=y, y_pred=y_pred)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
    return FinetuneModel(**kwargs)
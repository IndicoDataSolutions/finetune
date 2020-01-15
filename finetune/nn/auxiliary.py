import math

import tensorflow as tf	
from finetune.util.shapes import shape_list
from finetune.util.positional_embeddings import add_timing_signal_from_position
from finetune.base_models.gpt.featurizer import conv1d

def layer_norm_with_custom_init(input_tensor, name="", custom=False, pos_embed=None):
    """Run layer normalization on the last dimension of the tensor."""

    if custom:
        #scale mean and standard deviation
        mean = tf.math.reduce_mean(input_tensor)
        sd = tf.math.reduce_std(input_tensor)
        target_shape = shape_list(input_tensor)[1]
        weights = tf.get_variable(name+'gamma', shape=(target_shape - pos_embed))
        bias = tf.get_variable(name+'beta', shape=(target_shape - pos_embed))

        pos_weights = tf.get_variable(name+'pos_gamma', shape=(pos_embed))
        pos_bias = tf.get_variable(name+'pos_beta', shape=(pos_embed))

        full_weights = tf.concat((weights, pos_weights), axis=0)
        full_bias = tf.concat((bias, pos_bias), axis=0)

        return ((input_tensor-mean)/sd)*full_weights + full_bias
    else:
        return tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def dense_with_custom_init(input_tensor,
                           output_dim,
                           activation,
                           kernel_initializer,
                           name="",
                           custom=False,
                           pos_embed=None):

    if custom:
        # Subtracting pos_embed. input_tensor already includes context, and we
        # want separate weights for the words and the positional context
        original_weights = tf.get_variable(name+'/kernel',shape=(shape_list(input_tensor)[1]-pos_embed, output_dim-pos_embed))
        position_weights = tf.get_variable(name+"/pos_weights",
                                           shape=(pos_embed, pos_embed))

        original_weights = tf.pad(original_weights, tf.constant([[0,pos_embed], [0,0]]))
        # Note: Need to keep the dimension of original_weights before we pad it. Below
        # we use output_dim, put if we want to have a non-square matrix for original_weights
        # we should saving the first dimension of original_weights
        position_weights = tf.pad(position_weights, tf.constant([[shape_list(input_tensor)[1]-pos_embed,0], [0,0]]))
        # This concat should blow up (or give the wrong dimensions)
        full_weights = tf.concat((original_weights, position_weights), axis=1)


        original_bias = tf.get_variable(name+'/bias', shape=(output_dim-pos_embed))
        # Also using output_dim here in lieu of shape_list(original_weights)[0]
        # If we did that, it would be pos_embed + pos_embed + output_dim
        position_bias = tf.get_variable(name+"/pos_bias", shape=(pos_embed))
        full_bias = tf.concat((original_bias, position_bias), axis=0)
        return tf.matmul(input_tensor, full_weights) + full_bias

    else:
        return tf.layers.dense(input_tensor, output_dim, activation, name, kernel_initializer=kernel_initializer)



def embed_context(context, featurizer_state, config, train):
    with tf.variable_scope("context_embedding"):
        context_dim = shape_list(context)[-1]
        context_weight = tf.get_variable(
            name="ce",
            shape=[context_dim, config.n_context_embed],
            initializer=tf.random_normal_initializer(stddev=config.context_embed_stddev),
        )
        context_bias = tf.get_variable(
            name="ca",
            shape=[config.n_context_embed],
            initializer=tf.zeros_initializer(),
        )
        c_embed = tf.add(tf.tensordot(context, context_weight, axes=[[-1], [0]]), context_bias)
    featurizer_state['context'] = c_embed
    return featurizer_state


def add_context_embed(featurizer_state):
    if "context" in featurizer_state:
        context_embed = featurizer_state["context"]

        shape = shape_list(context_embed)
        if len(shape) == 4:
            # comparison / multiple choice 
            flat_embed = tf.reshape(
                context_embed, 
                [shape[0] * shape[1], shape[2], shape[3]],
            )
        else:
            flat_embed = context_embed

        seq_mask = tf.sequence_mask(featurizer_state['lengths'])
        for key in ['features', 'explain_out']:
            if key in featurizer_state:
                float_mask = tf.cast(seq_mask, tf.float32)
                binary_mask = tf.constant(1.) - float_mask
                flat_embed = flat_embed * tf.expand_dims(binary_mask, -1)
                sum_context = tf.reduce_sum(flat_embed, 1)
                mean_context = sum_context / tf.reduce_sum(float_mask)

                if len(shape) == 4:
                    mean_context = tf.reshape(
                        mean_context, 
                        [shape[0], shape[1], shape[3]]
                    )
    
                featurizer_state[key] = tf.concat(
                    (featurizer_state[key], mean_context), -1
                )

        featurizer_state['sequence_features'] = tf.concat(
            (featurizer_state['sequence_features'], context_embed), -1
        )


def embed_position(context, config, batch, seq):
    with tf.variable_scope("context_embedding"):
        context_dim = shape_list(context)[-1]
        context_channels = config.n_context_embed_per_channel * context_dim
        x = tf.zeros(shape=(batch, seq, context_channels))
        pos_embed = add_timing_signal_from_position(
            x,
            context,
            timescales = [
                [
                    (math.pi / 2) * (1/2500),
                    (25 * math.pi) * (1/2500)
                ]
            ] * context_dim
        ) / (float(context_channels) / config.context_embed_scale)
    return pos_embed


def add_context_embed(featurizer_state):
    if "context" in featurizer_state:
        context_embed = featurizer_state["context"]

        shape = shape_list(context_embed)
        if len(shape) == 4:
            # comparison / multiple choice
            flat_embed = tf.reshape(
                context_embed,
                [shape[0] * shape[1], shape[2], shape[3]],
            )
        else:
            flat_embed = context_embed

        seq_mask = tf.sequence_mask(featurizer_state['lengths'])
        for key in ['features', 'explain_out']:
            if key in featurizer_state:
                float_mask = tf.cast(seq_mask, tf.float32)
                binary_mask = tf.constant(1.) - float_mask
                flat_embed = flat_embed * tf.expand_dims(binary_mask, -1)
                sum_context = tf.reduce_mean(flat_embed, 1)
                mean_context = sum_context / tf.reduce_mean(float_mask)

                if len(shape) == 4:
                    mean_context = tf.reshape(
                        mean_context,
                        [shape[0], shape[1], shape[3]]
                    )

                featurizer_state[key] = tf.concat(
                    (featurizer_state[key], mean_context), -1
                )

        featurizer_state['sequence_features'] = tf.concat(
            (featurizer_state['sequence_features'], context_embed), -1
        )

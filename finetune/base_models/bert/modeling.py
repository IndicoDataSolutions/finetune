# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

import copy
import math
import numpy as np
from finetune.nn.nn_utils import ExtraScope, saver_ignore_scope
from finetune.base_models.bert.roberta_encoder import RoBERTaEncoder
import tensorflow as tf
import functools

from transformers.activations_tf import gelu as hf_gelu

from finetune.nn.auxiliary import embed_position


class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        vocab_size,
        hidden_size,
        initializer_range,
        embedding_post_processor,
        use_one_hot_embeddings=False,
        name="embeddings",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.embedding = EmbeddingLookup(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            initializer_range=initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings,
        )
        self.embedding_post_processor = embedding_post_processor

    def call(self, input_ids, input_context=None, token_type_ids=None):
        embedding_output = self.embedding(input_ids)
        return self.embedding_post_processor(
            input_tensor=embedding_output,
            input_context=input_context,
            token_type_ids=token_type_ids,
        )


@saver_ignore_scope
class DocRepPosEmbed(tf.keras.layers.Layer):
    def __init__(self, positional_channels, width, max_2d_positional_embeddings=None,**kwargs):
        # max_2d_positional_embeddings is unused but needed to keep the APIs consistent
        # **kwargs needs to be sent to keras layer
        super().__init__(**kwargs)
        self.positional_channels = positional_channels
        self.width = width
        self.dense1 = tf.keras.layers.Dense(
            width,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=0.02, mode="fan_avg", distribution="truncated_normal"
            )
        )

    def call(self, input_context):
        input_shape = tf.shape(input_context)
        return self.dense1(
            embed_position(
                input_context,
                self.positional_channels,
                input_shape[0],
                input_shape[1],
                self.width
            )
        )

@saver_ignore_scope
class LayoutLMPosEmbed(tf.keras.layers.Layer):
    def __init__(self, positional_channels, width, max_2d_positional_embeddings=1024, **kwargs):
        super().__init__(**kwargs)
        self.positional_channels = positional_channels
        self.width = width
        self.max_2d_positional_embeddings = max_2d_positional_embeddings
        
    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal()
        self.x_position_embedding_table = self.add_weight(
            name="x_position_embeddings",
            shape=[self.max_2d_positional_embeddings, self.width],
            initializer=initializer,
        )
        self.y_position_embedding_table = self.add_weight(
            name="y_position_embeddings",
            shape=[self.max_2d_positional_embeddings, self.width],
            initializer=initializer,
        )
        self.h_position_embedding_table = self.add_weight(
            name="h_position_embeddings",
            shape=[self.max_2d_positional_embeddings, self.width],
            initializer=initializer,
        )
        self.w_position_embedding_table = self.add_weight(
            name="w_position_embeddings",
            shape=[self.max_2d_positional_embeddings, self.width],
            initializer=initializer,
        )

    def call(self, input_context):
        bottom_pos = tf.cast(input_context[:, :, 0], dtype=tf.int32)
        left_pos = tf.cast(input_context[:, :, 1], dtype=tf.int32)
        right_pos = tf.cast(input_context[:, :, 2], dtype=tf.int32)
        top_pos = tf.cast(input_context[:, :, 3], dtype=tf.int32)
        
        left_position_embeddings = tf.gather(self.x_position_embedding_table, left_pos)
        upper_position_embeddings = tf.gather(self.y_position_embedding_table, top_pos)
        right_position_embeddings = tf.gather(self.x_position_embedding_table, right_pos)
        lower_position_embeddings = tf.gather(self.y_position_embedding_table, bottom_pos)
        h_position_embeddings = tf.gather(self.h_position_embedding_table, bottom_pos - top_pos)
        w_position_embeddings = tf.gather(self.w_position_embedding_table, right_pos - left_pos)
        all_2d_pos_embeddings = [
            left_position_embeddings,
            upper_position_embeddings,
            right_position_embeddings,
            lower_position_embeddings,
            h_position_embeddings,
            w_position_embeddings,
        ]
        return tf.math.add_n(all_2d_pos_embeddings)
        
@saver_ignore_scope
class XDocPosEmbed(tf.keras.layers.Layer):
    def __init__(self, positional_channels, width, **kwargs):
        super().__init__(**kwargs)
        self.positional_channels = positional_channels
        self.width = width
        self.layout_lm_pos_embed = LayoutLMPosEmbed(positional_channels, width)
        self.dense1 = tf.keras.layers.Dense(
            width,
            activation=tf.nn.relu,
            name="doc_linear1"
        )
        self.dense2 = tf.keras.layers.Dense(
            width,
            name="doc_linear2"
        )
        
    def call(self, input_context):
        layoutlm_pos = self.layout_lm_pos_embed(input_context)
        layoutlm_pos = self.dense1(layoutlm_pos)
        layoutlm_pos = self.dense2(layoutlm_pos)
        return layoutlm_pos

class BaseBertModel(tf.keras.layers.Layer):
    """BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

        ```python
        # Already been converted into WordPiece token ids
        input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
        input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
        token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

        config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

        model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

        label_embeddings = tf.get_variable(...)
        pooled_output = model.get_pooled_output()
        logits = tf.matmul(pooled_output, label_embeddings)
        ...
        ```
    """

    def __init__(
        self, encoder, config, pos2d_embedding_layer, token_type_vocab_size=None, name="bert", **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        config = copy.deepcopy(config)
        is_roberta = config.base_model.is_roberta
        is_roberta_v1 = is_roberta and config.base_model.encoder == RoBERTaEncoder
        self.max_position_embeddings = config.max_length
        self.vocab_size = encoder.vocab_size
        if is_roberta:
            self.max_position_embeddings += 2
        if is_roberta_v1:
            self.vocab_size += 1
        self.delimiter_token = encoder.delimiter_token
        self.use_token_type = config.bert_use_type_embed
        self.use_pooler = config.bert_use_pooler
        self.embed_dim = config.n_embed
        self.embeddings = Embedding(
            hidden_size=self.embed_dim,
            initializer_range=config.weight_stddev,
            vocab_size=self.vocab_size,
            embedding_post_processor=EmbeddingPostprocessor(
                feature_dim=self.embed_dim,
                use_token_type=self.use_token_type,
                token_type_vocab_size=2 or token_type_vocab_size, # Not sure why this is 2 but that's what finetune previously set it to.
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=not config.reading_order_removed,
                position_embedding_name="position_embeddings",
                initializer_range=config.weight_stddev,
                max_position_embeddings=self.max_position_embeddings,
                dropout_prob=config.resid_p_drop,
                roberta=is_roberta,
                pos_injection=config.context_injection,
                positional_channels=config.context_channels,
                pos2d_embedding_layer=pos2d_embedding_layer
            ),
            use_one_hot_embeddings=False,
        )

        self.transformer_model = TransformerModel(
            num_hidden_layers=config.n_layer,
            num_attention_heads=config.n_heads,
            intermediate_size=config.bert_intermediate_size,
            intermediate_act_fn=get_activation(config.act_fn),
            hidden_dropout_prob=config.resid_p_drop,
            attention_probs_dropout_prob=config.attn_p_drop,
            initializer_range=config.weight_stddev,
            recompute_grad=config.low_memory_mode,
        )

        if self.use_pooler:
            self.pooler = tf.keras.layers.Dense(
                self.embed_dim,
                activation=tf.tanh,
                kernel_initializer=create_initializer(config.initializer_range),
            )
        
    def call(self, tokens, context, sequence_lengths, training=True):
        
        delimiters = tf.cast(tf.equal(tokens, self.delimiter_token), tf.int32)
        token_type_ids = tf.cumsum(delimiters, exclusive=True, axis=1)
        input_shape = get_shape_list(tokens, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        input_mask = tf.sequence_mask(sequence_lengths, maxlen=seq_length, dtype=tf.float32)

        embedding_output = self.embeddings(
            input_ids=tokens,
            input_context=context,
            token_type_ids=token_type_ids,
        )
        
        attention_mask = create_attention_mask_from_input_mask(
            tokens, input_mask
        )

        sequence_output = self.transformer_model(
            layer_input=embedding_output,
            batch_size=batch_size,
            seq_length=seq_length,
            attention_mask=attention_mask,
        )

        output_shape = tf.shape(sequence_output)

        def first_token():
            return tf.squeeze(sequence_output[:, 0:1, :], axis=1)

        def empty():
            return tf.zeros(
                tf.concat([[output_shape[0]], [self.embed_dim]], axis=0),
                dtype=sequence_output.dtype,
            )

        pooled_output = tf.cond(
            tf.equal(output_shape[1], 0), true_fn=empty, false_fn=first_token
        )
        pooled_output.set_shape([None, self.embed_dim])
        if self.use_pooler:
            pooled_output = self.pooler(pooled_output)

        return {
            "sequence_features": sequence_output,
            "features": pooled_output,
        }
    

def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.

    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(activation_string):
    """
    Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, str):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    elif act == "hf_gelu":
        return hf_gelu
    else:
        raise ValueError("Unsupported activation: %s" % act)


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, name="LayerNorm", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )

    def call(self, input_tensor):
        inputs_shape = input_tensor.shape
        inputs_rank = inputs_shape.ndims
        begin_norm_axis = inputs_rank - 1
        # Calculate the moments on the last axis (layer activations).
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = tf.nn.moments(input_tensor, norm_axes, keepdims=True)
        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1e-12 if self.compute_dtype != tf.float16 else 1e-3
        outputs = tf.nn.batch_normalization(
            input_tensor,
            mean,
            variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=variance_epsilon,
        )
        outputs.set_shape(inputs_shape)
        return outputs

@saver_ignore_scope
class LayerNormAndDropout(tf.keras.layers.Layer):
    def __init__(self, dropout_prob, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNorm()
        self.dropout = tf.keras.layers.Dropout(dropout_prob)

    def call(self, input_tensor):
        return self.dropout(self.layer_norm(input_tensor))


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


@saver_ignore_scope
class EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size=128, initializer_range=0.02, word_embedding_name="word_embeddings", use_one_hot_embeddings=False, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.word_embedding_name = word_embedding_name
        self.use_one_hot_embeddings = use_one_hot_embeddings

    def build(self, input_shape):
        self.embedding_table = self.add_weight(
            name=self.word_embedding_name,
            shape=[self.vocab_size, self.embedding_size],
            initializer=create_initializer(self.initializer_range),
        )

    def call(self, input_ids):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])

        flat_input_ids = tf.reshape(input_ids, [-1])
        if self.use_one_hot_embeddings:
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
            output = tf.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)

        input_shape = get_shape_list(input_ids)

        output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output


@saver_ignore_scope
class EmbeddingPostprocessor(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            use_token_type=False,
            token_type_vocab_size=16,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=0.02,
            max_position_embeddings=512,
            dropout_prob=0.1,
            roberta=False,
            pos_injection=False,
            positional_channels=None,
            pos2d_embedding_layer=DocRepPosEmbed,
            feature_dim=None,
            max_2d_positional_embeddings=None,
            **kwargs):
        super().__init__(**kwargs)
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.token_type_embedding_name = token_type_embedding_name
        self.use_position_embeddings = use_position_embeddings
        self.position_embedding_name = position_embedding_name
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.roberta = roberta
        self.pos_injection = pos_injection
        self.positional_channels = positional_channels
        self.feature_dim = feature_dim
        self.pos2d_embedding_layer = pos2d_embedding_layer(
            positional_channels=positional_channels,
            width=feature_dim,
            max_2d_positional_embeddings=max_2d_positional_embeddings,
        )
        self.layer_norm_and_dropout = LayerNormAndDropout(dropout_prob)
        if self.pos_injection:
            self.pos2d_embedding = self.pos2d_embedding_layer(self.positional_channels, feature_dim)

        
    def build(self, input_shape):
        if self.use_token_type:
            self.token_type_table = self.add_weight(
                name=self.token_type_embedding_name,
                shape=[self.token_type_vocab_size, self.feature_dim],
                initializer=create_initializer(self.initializer_range),
            )

        if self.use_position_embeddings:
            self.position_embedding_table = self.add_weight(
                name=self.position_embedding_name,
                shape=[self.max_position_embeddings, self.feature_dim],
                initializer=create_initializer(self.initializer_range),
            )


    def call(self, input_tensor, input_context, token_type_ids, position_ids=None):
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        output = input_tensor

        if self.use_token_type:
            if token_type_ids is None:
                raise ValueError(
                    "`token_type_ids` must be specified if" "`use_token_type` is True."
                )
            # This vocab will be small so we always do one-hot here, since it is always
            # faster for a small vocabulary.
            token_type_table = tf.convert_to_tensor(value=self.token_type_table)
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            one_hot_ids = tf.one_hot(
                flat_token_type_ids,
                depth=self.token_type_vocab_size,
                dtype=self.token_type_table.dtype,
            )
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(
                token_type_embeddings, [batch_size, seq_length, width]
            )
            output += token_type_embeddings

        if self.use_position_embeddings:
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            if self.roberta:
                position_embeddings = self.position_embedding_table[2:seq_length + 2]
            else:
                position_embeddings = self.position_embedding_table[:seq_length]

            if position_ids is None:
                num_dims = len(output.shape.as_list())
                # Only the last two dimensions are relevant (`seq_length` and `width`), so
                # we broadcast among the first dimensions, which is typically just
                # the batch size.
                position_broadcast_shape = []
                for _ in range(num_dims - 2):
                    position_broadcast_shape.append(1)
                position_broadcast_shape.extend([seq_length, width])
                position_embeddings = tf.reshape(
                    position_embeddings, position_broadcast_shape
                )
            else:
                position_embeddings = tf.gather(position_embeddings, position_ids)
            output += position_embeddings

        if self.pos_injection and input_context is not None:
            output += self.pos2d_embedding(
                input_context=input_context,
            )
        output = self.layer_norm_and_dropout(output)
        return output



def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=to_mask.dtype
    )

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=to_mask.dtype
    )

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        num_attention_heads=1,
        size_per_head=512,
        query_act=None,
        key_act=None,
        value_act=None,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        do_return_2d_tensor=False,
        name="self",
         **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.do_return_2d_tensor = do_return_2d_tensor

        self.query_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(initializer_range),
        )

        self.key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(initializer_range),
        )

        self.value_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(initializer_range),
        )

        self.attention_probs_dropout = tf.keras.layers.Dropout(attention_probs_dropout_prob)

    def call(self, from_tensor, to_tensor, attention_mask=None, batch_size=None, from_seq_length=None, to_seq_length=None):
        def transpose_for_scores(
            input_tensor, batch_size, num_attention_heads, seq_length, width
        ):
            output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width]
            )

            output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
            return output_tensor

        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`."
            )

        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if batch_size is None or from_seq_length is None or to_seq_length is None:
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified."
                )

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        # `query_layer` = [B*F, N*H]
        query_layer = self.query_layer(from_tensor_2d)

        # `key_layer` = [B*T, N*H]
        key_layer = self.key_layer(to_tensor_2d)

        # `value_layer` = [B*T, N*H]
        value_layer = self.value_layer(to_tensor_2d)

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(
            query_layer, batch_size, self.num_attention_heads, from_seq_length, self.size_per_head
        )

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(
            key_layer, batch_size, self.num_attention_heads, to_seq_length, self.size_per_head
        )

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(
            attention_scores, 1.0 / math.sqrt(float(self.size_per_head))
        )

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_probs_dropout(attention_probs)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer, [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head]
        )

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])

        if self.do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head],
            )
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head],
            )

        return context_layer


class Attention(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            attention_head_size,
            num_attention_heads=12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            name="attention",
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.attention_layer = AttentionLayer(
            size_per_head=attention_head_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=True,
        )

        self.output_dropout = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
        self.output_layer_norm = ExtraScope(LayerNorm(), "output")
        self.initializer_range = initializer_range

    def build(self, input_shape):
        # We could clean this up by adding an output layer with this and the layernorm?
        self.output_layer = ExtraScope(tf.keras.layers.Dense(
            input_shape[-1],
            name="dense",
            kernel_initializer=create_initializer(self.initializer_range),
        ), "output")

    def call(self, *, layer_input, batch_size, seq_length, attention_mask=None):
        attention_output = self.attention_layer(
            from_tensor=layer_input,
            to_tensor=layer_input,
            attention_mask=attention_mask,
            batch_size=batch_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
        )
        attention_output = self.output_layer(attention_output)
        attention_output = self.output_dropout(attention_output)
        attention_output = self.output_layer_norm(attention_output + layer_input)
        return attention_output

class FullBlock(tf.keras.layers.Layer):
    def __init__(self,
        *,
        attention_head_size,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn=tf.nn.gelu,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        name=None,
        **kwargs
    ):
        assert name is not None
        super().__init__(name=name, **kwargs)
        self.attention = Attention(
            attention_head_size=attention_head_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
        )
        self.intermediate = ExtraScope(
            tf.keras.layers.Dense(
                intermediate_size,
                activation=intermediate_act_fn,
                kernel_initializer=create_initializer(initializer_range),
                name="dense",
            ),
            "intermediate"
        )
        self.output_dropout = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
        self.output_layer_norm = ExtraScope(LayerNorm(), "output")
        self.initializer_range = initializer_range
    
    def build(self, input_shape):
        self.output_layer = ExtraScope(
            tf.keras.layers.Dense(
                input_shape[-1],
                kernel_initializer=create_initializer(self.initializer_range),
                name="dense",
            ),
            "output"
        )

    def call(self, layer_input, batch_size, seq_length, attention_mask=None):
        attention_output = self.attention(
            layer_input=layer_input,
            batch_size=batch_size,
            seq_length=seq_length,
            attention_mask=attention_mask,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output_layer(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        return layer_output

class TransformerModel(tf.keras.layers.Layer):
    def __init__(self,
        *,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn=gelu,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        recompute_grad=False,
        name="encoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.recompute_grad = recompute_grad
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        if hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, self.num_attention_heads)
                )

        attention_head_size = int(hidden_size / self.num_attention_heads)
        self.blocks = [
            FullBlock(
                attention_head_size=attention_head_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                intermediate_act_fn=self.intermediate_act_fn,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                initializer_range=self.initializer_range,
                name=f"layer_{i}",
            )
            for i in range(self.num_hidden_layers)
        ]

    def call(self, *, layer_input, batch_size, seq_length, attention_mask=None, training=False):
        input_shape = get_shape_list(layer_input, expected_rank=3)
        prev_output = reshape_to_matrix(layer_input)

        for i, block in enumerate(self.blocks):
            if self.recompute_grad and training:
                block = tf.recompute_grad(block)
            prev_output = block(layer_input=prev_output, batch_size=batch_size, seq_length=seq_length, attention_mask=attention_mask)
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output



def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
            be returned as python integers, and dynamic dimensions will be returned
            as tf.Tensor scalars.
    """

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError(
            "Input tensor must have at least rank 2. Shape = %s" % (input_tensor.shape)
        )
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def twin_transformer_model(
    input_tensor_a,
    input_tensor_b,
    attention_mask_a=None,
    attention_mask_b=None,
    mixing_inputs=None,
    mixing_fn=None,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    intermediate_act_fn=gelu,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    initializer_range=0.02,
    do_return_all_layers=False,
    low_memory_mode=False,
):
    assert do_return_all_layers == False
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads)
        )

    if mixing_inputs is None:
        mixing_inputs = dict()

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape_a = get_shape_list(input_tensor_a, expected_rank=3)
    input_shape_b = get_shape_list(input_tensor_b, expected_rank=3)

    input_width = input_shape_a[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError(
            "The width of the input tensor (%d) != hidden size (%d)"
            % (input_width, hidden_size)
        )

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output_a = reshape_to_matrix(input_tensor_a)
    prev_output_b = reshape_to_matrix(input_tensor_b)

    for layer_idx in range(num_hidden_layers):
        with tf.compat.v1.variable_scope("layer_a_%d" % layer_idx):
            block_fn = functools.partial(
                full_block,
                attention_head_size=attention_head_size,
                batch_size=input_shape_a[0],
                seq_length=input_shape_a[1],
                attention_mask=attention_mask_a,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                intermediate_act_fn=intermediate_act_fn,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
            )

            if low_memory_mode:
                block_fn = recompute_grad(block_fn, use_entire_scope=True)
            prev_output_a = block_fn(prev_output_a)

        with tf.compat.v1.variable_scope("layer_b_%d" % layer_idx):
            block_fn = functools.partial(
                full_block,
                attention_head_size=attention_head_size,
                batch_size=input_shape_b[0],
                seq_length=input_shape_b[1],
                attention_mask=attention_mask_b,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                intermediate_act_fn=intermediate_act_fn,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
            )

            if low_memory_mode:
                block_fn = recompute_grad(block_fn, use_entire_scope=True)
            prev_output_b = block_fn(prev_output_b)

        if mixing_fn is not None and layer_idx != num_attention_heads - 1:
            # TODO: make this configurable - how many cross adaptors we include.
            if layer_idx % 2 == 0:
                with tf.compat.v1.variable_scope("mixing_fn_%d" % layer_idx):
                    mix_output_a, mix_output_b = mixing_fn(
                        reshape_from_matrix(prev_output_a, input_shape_a),
                        reshape_from_matrix(prev_output_b, input_shape_b),
                        **mixing_inputs
                    )
                    prev_output_a = reshape_to_matrix(mix_output_a)
                    prev_output_b = reshape_to_matrix(mix_output_b)

    final_output_a = reshape_from_matrix(prev_output_a, input_shape_a)
    final_output_b = reshape_from_matrix(prev_output_b, input_shape_b)

    return final_output_a, final_output_b


class TwinBertModel():#_BertModel):
    def __init__(
        self,
        config,
        is_training,
        input_ids_a,
        input_ids_b,
        attention_mask_a=None,
        attention_mask_b=None,
        token_type_ids_a=None,
        token_type_ids_b=None,
        context_a=None,
        context_b=None,
        pos_ids_a=None,
        pos_ids_b=None,
        mixing_fn=None,
        mixing_inputs=None,
        use_one_hot_embeddings=False,
        scope=None,
        roberta=False,
        use_token_type=True,
        reading_order_decay_rate=None,
    ):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings.
            scope: (optional) variable scope. Defaults to "bert".

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        self.table_position_type = config.table_position_type
        self.max_row_col_embedding = config.max_row_col_embedding
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            with tf.compat.v1.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output_a, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids_a,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings,
                )
                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output_a = self.embedding_postprocessor(
                    input_tensor=self.embedding_output_a,
                    input_context=context_a,
                    use_token_type=use_token_type,
                    token_type_ids=token_type_ids_a,
                    position_ids=pos_ids_a,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=not config.reading_order_removed,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob,
                    roberta=roberta,
                    positional_channels=config.positional_channels,
                    reading_order_decay_rate=reading_order_decay_rate,
                    anneal_reading_order=config.anneal_reading_order,
                    pos_injection=config.table_position,  # True,
                )

            with tf.compat.v1.variable_scope("embeddings", reuse=True):
                # Perform embedding lookup on the word ids.
                (self.embedding_output_b, _) = embedding_lookup(
                    input_ids=input_ids_b,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings,
                )
                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output_b = self.embedding_postprocessor(
                    input_tensor=self.embedding_output_b,
                    input_context=context_b,
                    use_token_type=use_token_type,
                    token_type_ids=token_type_ids_b,
                    position_ids=pos_ids_b,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=not config.reading_order_removed,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob,
                    roberta=roberta,
                    positional_channels=config.positional_channels,
                    reading_order_decay_rate=reading_order_decay_rate,
                    anneal_reading_order=config.anneal_reading_order,
                    pos_injection=config.table_position,  # True,
                )

            with tf.compat.v1.variable_scope("encoder"):
                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.sequence_output = twin_transformer_model(
                    input_tensor_a=self.embedding_output_a,
                    input_tensor_b=self.embedding_output_b,
                    attention_mask_a=attention_mask_a,
                    attention_mask_b=attention_mask_b,
                    mixing_inputs=mixing_inputs,
                    mixing_fn=mixing_fn,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=False,
                    low_memory_mode=config.low_memory_mode and is_training,
                )
                self.all_encoder_layers = None

    def embedding_postprocessor(self, *args, **kwargs):
        def table_pos_embed(
            input_context, positional_channels, batch_size, seq_length, width
        ):
            output = []
            for entry in (
                [0, 1] if self.table_position_type == "row_col" else [0, 1, 2, 3]
            ):
                position_table = tf.compat.v1.get_variable(
                    name="pos_{}".format(entry),
                    shape=[self.max_row_col_embedding, width],
                    initializer=tf.compat.v1.random_normal_initializer(stddev=1e-3),
                )
                position = tf.cast(input_context[:, :, entry], dtype="int32")
                output.append(tf.gather(position_table, position))
            return tf.math.add_n(output)

        return embedding_postprocessor(
            *args, pos2d_embedding_fn=table_pos_embed, **kwargs
        )


class BertModel(tf.keras.layers.Layer):
    def __init__(self, encoder, config, **kwargs):
        super().__init__(**kwargs)
        self.base_bert = BaseBertModel(
            encoder=encoder,
            config=config,
            pos2d_embedding_layer=DocRepPosEmbed
        )

    def call(self, tokens, context, sequence_lengths):
        return self.base_bert(
            tokens=tokens,
            context=context,
            sequence_lengths=sequence_lengths,
        )


class LayoutLMModel(tf.keras.layers.Layer):
    def __init__(self, encoder, config, **kwargs):
        super().__init__(**kwargs)
        self.base_bert = BaseBertModel(
            encoder=encoder,
            config=config,
            pos2d_embedding_layer=LayoutLMPosEmbed
        )

    def call(self, tokens, context, sequence_lengths):
        return self.base_bert(
            tokens=tokens,
            context=context,
            sequence_lengths=sequence_lengths,
        )

class XDocModel(tf.keras.layers.Layer):
    def __init__(self, encoder, config, **kwargs):
        super().__init__(**kwargs)
        self.base_bert = BaseBertModel(
            encoder=encoder,
            config=config,
            token_type_vocab_size=1,
            pos2d_embedding_layer=XDocPosEmbed
        )

    def call(self, tokens, context, sequence_lengths):
        return self.base_bert(
            tokens=tokens,
            context=context,
            sequence_lengths=sequence_lengths,
        )

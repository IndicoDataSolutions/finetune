import os
import tensorflow as tf
import finetune
import functools

from typing import Optional, Tuple, Union
import json


FINETUNE_FOLDER = os.path.dirname(finetune.__file__)
CONFIG_PATH = os.path.join(FINETUNE_FOLDER, "model", "modern_bert", "config.json")

class ModernBertConfig:
    # TODO: make this configurable so we can support different models.
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)

    def __getattr__(self, name):
        return self.config[name]


class EmbeddingWithPadIdx(tf.keras.layers.Embedding):
    # TODO: we are not masking grads for the pad idx like the torch version.
    # Shouldn't be necessary as there should be no gradient here anyway.
    # But worth looking at if we run into issues
    def __init__(self, *args, pad_idx=None, name="EmbeddingWithPadIdx", **kwargs):
        super().__init__(*args, **kwargs, name=name)
        self.pad_idx = pad_idx

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, self.pad_idx)


class ModernBertEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: ModernBertConfig, name="Embedding"):
        super().__init__(name=name)
        self.config = config
        self.tok_embeddings = EmbeddingWithPadIdx(config.vocab_size, config.hidden_size, pad_idx=config.pad_token_id)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=config.norm_eps, center=config.norm_bias, name="EmbeddingNorm")
        self.drop = tf.keras.layers.Dropout(config.embedding_dropout)

    def call(
        self, input_ids: tf.Tensor = None, inputs_embeds: Optional[tf.Tensor] = None, training=False
    ) -> tf.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds), training=training)
        else:
            hidden_states = self.drop(self.norm(self.tok_embeddings(input_ids)), training=training)
        return hidden_states


class ModernBertMLP(tf.keras.layers.Layer):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: ModernBertConfig, name="GLU"):
        super().__init__(name=name)
        self.config = config
        self.Wi = tf.keras.layers.Dense(int(config.intermediate_size) * 2, use_bias=config.mlp_bias, name="Wi")
        if config.hidden_activation == "gelu":
            self.act = functools.partial(tf.keras.activations.gelu, approximate=False)
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_activation}")
        self.drop = tf.keras.layers.Dropout(config.mlp_dropout)
        self.Wo = tf.keras.layers.Dense(config.hidden_size, use_bias=config.mlp_bias, name="Wo")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        input, gate = tf.split(self.Wi(hidden_states), 2, axis=-1)
        result = self.Wo(self.drop(self.act(input) * gate))
        return result


class ModernBertRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (tf.range(0, dim, 2, dtype=tf.float32) / dim))

    def call(self, x, position_ids):
        inv_freq_expanded = tf.tile(tf.expand_dims(tf.expand_dims(self.inv_freq, 0), 2), [position_ids.shape[0], 1, 1])
        position_ids_expanded = tf.expand_dims(position_ids, 1)
        freqs = tf.transpose(tf.matmul(inv_freq_expanded, position_ids_expanded), perm=[0, 2, 1])
        emb = tf.concat([freqs, freqs], axis=-1)
        cos = tf.cast(tf.cos(emb), x.dtype)
        sin = tf.cast(tf.sin(emb), x.dtype)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tf.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = tf.expand_dims(cos, unsqueeze_dim)
    sin = tf.expand_dims(sin, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed




class ModernBertAttention(tf.keras.layers.Layer):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    See `forward` method for additional details.
    """

    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None, name="Attention"):
        super().__init__(name=name)
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = tf.keras.layers.Dense(3 * self.all_head_size, use_bias=config.attention_bias, name="Wqkv")

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta

        self.rotary_emb = ModernBertRotaryEmbedding(dim=self.head_dim, base=rope_theta)

        self.Wo = tf.keras.layers.Dense(config.hidden_size, use_bias=config.attention_bias, name="Wo")
        self.out_drop = tf.keras.layers.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else tf.keras.layers.Identity()
        self.pruned_heads = set()

        
    def eager_attention_forward(
        self,
        qkv: tf.Tensor,
        attention_mask: tf.Tensor,
        sliding_window_mask: tf.Tensor,
        position_ids: Optional[tf.Tensor],
        local_attention: Tuple[int, int],
        bs: int,
        dim: int,
        training: bool,
    ) -> Union[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]:
        # qkv: [batch_size, seqlen, 3, nheads, headdim]
        cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        query, key, value = tf.unstack(tf.transpose(qkv, perm=[0, 3, 2, 1, 4]), 3, axis=2)
        # query, key, value: [batch_size, heads, seq_len, head_dim]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        scale = self.head_dim**-0.5
        attn_weights = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) * scale

        if local_attention != (-1, -1):
            attention_mask = sliding_window_mask

        attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = tf.keras.activations.softmax(attn_weights, axis=-1)
        attn_weights = tf.keras.layers.Dropout(rate=self.attention_dropout)(attn_weights, training=training)
        attn_output = tf.matmul(attn_weights, value)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [bs, -1, dim])
        return attn_output

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask,
        position_ids,
        sliding_window_mask,
        training: bool,
    ) -> tf.Tensor:
        qkv = self.Wqkv(hidden_states)
        bs = tf.shape(hidden_states)[0]
        qkv = tf.reshape(qkv, [bs, -1, 3, self.num_heads, self.head_dim])

        attn_outputs = self.eager_attention_forward(
            qkv=qkv,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            training=training,
        )
        hidden_states = self.out_drop(self.Wo(attn_outputs))
        return hidden_states

class ModernBertEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None, name="EncoderLayer"):
        super().__init__(name=name)
        self.config = config
        if layer_id == 0:
            self.attn_norm = tf.keras.layers.Identity()
        else:
            self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=config.norm_eps, center=config.norm_bias, name="AttnNorm")
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=config.norm_eps, center=config.norm_bias, name="MLPNorm")
        self.mlp = ModernBertMLP(config)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        sliding_window_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
       
        normed = self.attn_norm(hidden_states)
        attn_outputs = self.attn(
            normed,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attn_outputs
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return hidden_states



class ModernBert(tf.keras.layers.Layer):
    def __init__(self, config, name="ModernBert"):
        super().__init__(name=name)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [ModernBertEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=config.norm_eps, center=config.norm_bias, name="FinalNorm")

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        sliding_window_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        seq_len: Optional[int] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor, ...], dict]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        position_ids = tf.expand_dims(tf.range(seq_len, dtype=tf.float32), 0)

        attention_mask, sliding_window_mask = self._update_attention_mask(attention_mask)

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds, training=training)
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.final_norm(hidden_states)
        return hidden_states


    def _update_attention_mask(self, attention_mask: tf.Tensor) -> tf.Tensor:
        expanded_mask = tf.tile(attention_mask[:, None, None, :], [1, 1, tf.shape(attention_mask)[1], 1])
        inverted_mask = 1.0 - expanded_mask

        global_attention_mask = tf.where(tf.cast(inverted_mask, tf.bool), 
                                       tf.fill(tf.shape(inverted_mask), tf.float32.min), 
                                       inverted_mask)

        # Create position indices
        rows = tf.expand_dims(tf.range(tf.shape(global_attention_mask)[2]), 0)
        # Calculate distance between positions
        distance = tf.abs(rows - tf.transpose(rows))

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            tf.expand_dims(tf.expand_dims(distance <= self.config.local_attention // 2, 0), 0)
        )
        # Combine with existing mask
        sliding_window_mask = tf.where(tf.logical_not(window_mask),
                                     tf.fill(tf.shape(global_attention_mask), tf.float32.min),
                                     global_attention_mask)
        return global_attention_mask, sliding_window_mask

import tensorflow as tf
import functools
from finetune.optimizers.recompute_grads import recompute_grads_w_kwargs

from typing import Optional, Tuple, Union

class EmbeddingWithPadIdx(tf.keras.layers.Embedding):
    def __init__(self, *args, pad_idx=None, name="EmbeddingWithPadIdx", **kwargs):
        super().__init__(*args, **kwargs, name=name)
        self.pad_idx = pad_idx

class ModernBertEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: 'Settings', vocab_size: int, name="Embedding"):
        super().__init__(name=name)
        self.config = config
        self.tok_embeddings = EmbeddingWithPadIdx(
            vocab_size, config.n_embed, pad_idx=config.pad_idx
        )
        self.norm = tf.keras.layers.LayerNormalization(
            epsilon=config.norm_eps, center=False, name="EmbeddingNorm"
        )
        self.drop = tf.keras.layers.Dropout(config.embed_p_drop)

    def call(
        self,
        input_ids: tf.Tensor = None,
        training=False,
    ) -> tf.Tensor:
        embedded = self.tok_embeddings(input_ids)
        normed_embeddings = self.norm(embedded)
        dropped_embeddings = self.drop(normed_embeddings, training=training)
        return dropped_embeddings


class ModernBertMLP(tf.keras.layers.Layer):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: 'Settings', name="GLU"):
        super().__init__(name=name)
        self.config = config
        self.Wi = tf.keras.layers.Dense(
            int(config.bert_intermediate_size) * 2, use_bias=False, name="Wi"
        )
        self.act = functools.partial(tf.keras.activations.gelu, approximate=False)
        self.drop = tf.keras.layers.Dropout(config.mlp_p_drop)
        self.Wo = tf.keras.layers.Dense(
            config.n_embed, use_bias=False, name="Wo"
        )

    def call(self, hidden_states: tf.Tensor, training: bool) -> tf.Tensor:
        input, gate = tf.split(self.Wi(hidden_states), 2, axis=-1)
        result = self.Wo(self.drop(self.act(input) * gate, training=training))
        return result


class ModernBertRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.inv_freq = 1.0 / tf.pow(
            base, (tf.range(start=0, limit=dim, delta=2, dtype=tf.float32) / dim)
        )

    def call(self, position_ids):
        # 1, seq_len, 1
        inv_freq_expanded = self.inv_freq[None, :, None]
        # 1, 1, seq_len
        position_ids_expanded = position_ids[:, None, :]
        freqs = tf.transpose(
            tf.matmul(inv_freq_expanded, tf.cast(position_ids_expanded, tf.float32)), perm=[0, 2, 1]
        )
        emb = tf.concat([freqs, freqs], axis=-1)
        cos = tf.cos(emb)
        sin = tf.sin(emb)
        return tf.cast(cos, self.compute_dtype), tf.cast(sin, self.compute_dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tf.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`tf.Tensor`): The query tensor.
        k (`tf.Tensor`): The key tensor.
        cos (`tf.Tensor`): The cosine part of the rotary embedding.
        sin (`tf.Tensor`): The sine part of the rotary embedding.
        position_ids (`tf.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(tf.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
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

    def __init__(
        self, config: 'Settings', layer_id: Optional[int] = None, name="Attention"
    ):
        super().__init__(name=name)
        self.config = config
        self.layer_id = layer_id

        if config.n_embed % config.n_heads != 0:
            raise ValueError(
                f"The hidden size ({config.n_embed}) is not a multiple of the number of attention heads ({config.n_heads})"
            )

        self.attention_dropout = config.attn_p_drop
        self.num_heads = config.n_heads
        self.head_dim = config.n_embed // config.n_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = tf.keras.layers.Dense(
            3 * self.all_head_size, use_bias=False, name="Wqkv"
        )

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (
                config.local_attention_window // 2,
                config.local_attention_window // 2,
            )
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta

        self.rotary_emb = ModernBertRotaryEmbedding(dim=self.head_dim, base=rope_theta)

        self.Wo = tf.keras.layers.Dense(
            config.n_embed, use_bias=False, name="Wo"
        )
        self.out_drop = tf.keras.layers.Dropout(config.attn_p_drop)
        self.attn_drop = tf.keras.layers.Dropout(rate=self.attention_dropout)
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
        cos, sin = self.rotary_emb(position_ids=position_ids)
        query, key, value = tf.unstack(
            tf.transpose(qkv, perm=[0, 3, 2, 1, 4]), 3, axis=2
        )

        # query, key, value: [batch_size, heads, seq_len, head_dim]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        scale = self.head_dim**-0.5
        attn_weights = tf.matmul(query, key, transpose_b=True) * scale

        if local_attention != (-1, -1):
            attention_mask = sliding_window_mask

        attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = tf.keras.activations.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_drop(attn_weights, training=training)
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
        hidden_states = self.out_drop(self.Wo(attn_outputs), training=training)
        return hidden_states


class ModernBertEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        config: 'Settings',
        layer_id: Optional[int] = None,
        name="EncoderLayer",
    ):
        super().__init__(name=name)
        self.config = config
        self.layer_id = layer_id
        if layer_id == 0:
            self.attn_norm = tf.keras.layers.Identity()
        else:
            self.attn_norm = tf.keras.layers.LayerNormalization(
                epsilon=config.norm_eps, center=False, name="AttnNorm"
            )
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.norm_eps, center=False, name="MLPNorm"
        )
        self.mlp = ModernBertMLP(config)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        sliding_window_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        normed = self.attn_norm(hidden_states)
        attn_outputs = self.attn(
            normed,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            training=training,
        )
        hidden_states = hidden_states + attn_outputs
        mlp_output = self.mlp(self.mlp_norm(hidden_states), training=training)
        hidden_states = hidden_states + mlp_output

        return hidden_states


class ModernBert(tf.keras.layers.Layer):
    def __init__(self, config, vocab_size, name="ModernBert"):
        super().__init__(name=name)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config, vocab_size=vocab_size)
        self.layers = [
            ModernBertEncoderLayer(config, layer_id)
            for layer_id in range(config.n_layer)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.norm_eps, center=False, name="FinalNorm"
        )

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        seq_len: int,
        training: bool,
    ) -> Union[Tuple[tf.Tensor, ...], dict]:
        position_ids = tf.expand_dims(tf.range(seq_len, dtype=tf.float32), 0)
        attention_mask, sliding_window_mask = self._update_attention_mask(
            attention_mask
        )

        hidden_states = self.embeddings(input_ids=input_ids, training=training)
        for encoder_layer in self.layers:
            if self.config.low_memory_mode and training:
                encoder_layer.call = recompute_grads_w_kwargs(
                    encoder_layer.call,
                    train_vars=encoder_layer.trainable_weights,
                    name=encoder_layer.name
                )
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                training=training,
            )
            hidden_states = layer_outputs

        hidden_states = self.final_norm(hidden_states)
        return hidden_states

    def _update_attention_mask(self, attention_mask: tf.Tensor) -> tf.Tensor:
        attention_mask = tf.cast(attention_mask, self.compute_dtype)
        expanded_mask = tf.tile(
            attention_mask[:, None, None, :], [1, 1, tf.shape(attention_mask)[1], 1]
        )
        inverted_mask = 1.0 - expanded_mask
        ignore_value = tf.fill(tf.shape(inverted_mask), tf.cast(tf.float16.min, self.compute_dtype))

        global_attention_mask = tf.where(
            inverted_mask > 0.5, ignore_value, inverted_mask
        )
        # Create position indices
        rows = tf.expand_dims(tf.range(tf.shape(global_attention_mask)[2]), 0)
        # Calculate distance between positions
        distance = tf.abs(rows - tf.transpose(rows))

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = tf.expand_dims(
            tf.expand_dims(distance <= self.config.local_attention_window // 2, 0), 0
        )
        # Combine with existing mask
        sliding_window_mask = tf.where(window_mask, global_attention_mask, ignore_value)
        return global_attention_mask, sliding_window_mask

import functools
import tensorflow as tf
from finetune.optimizers.recompute_grads import recompute_grads_w_kwargs


def merge_output_state(states, X, lengths, chunk_pos_embed, hidden_dim):
    if chunk_pos_embed and chunk_pos_embed.lower() == "learned":
        pos_embed = tf.compat.v1.get_variable(
            name="chunk_pos_embed",
            shape=[len(states), hidden_dim],
            initializer=tf.compat.v1.random_normal_initializer(stddev=0.001),
        )
        for i, state in enumerate(states):
            state["sequence_features"] = state["sequence_features"] + tf.reshape(pos_embed[i], [1, 1, hidden_dim])
    else:
        assert chunk_pos_embed is None, "Only None and learned chunk_pos_embed are supported"
    output_state = {
        "features": tf.reduce_mean([s["features"] for s in states]),
        "sequence_features": tf.concat([s["sequence_features"] for s in states], 1),
        "lengths": lengths,
        "inputs": X,
    }
    if "decoder" in states[0]:
        output_state["decoder"] = states[0]["decoder"]
    if "embedding" in states[0]:
        output_state["embedding"] = states[0]["embedding"]
    if "embed_weights" in states[0]:
        output_state["embed_weights"] = states[0]["embed_weights"]
    return output_state


def fused_featurizer(featurizer):

    featurizer_w_recompute = recompute_grads_w_kwargs(featurizer, use_entire_scope=True)
   
    @functools.wraps(featurizer)
    def internal(X, encoder, config, train, reuse=None, lengths=None, **kwargs):
        if config.num_fusion_shards is None or config.num_fusion_shards == 1:
            return featurizer(X, encoder, config, train, reuse=reuse, lengths=lengths, **kwargs)
        orig_lengths = lengths
        out = []
        subsize = config.max_length // config.num_fusion_shards
        context = kwargs.pop("context")
        for start in range(0, config.max_length, subsize):
            xi = X[:, start : start + subsize]
            lengths = tf.maximum(lengths - tf.shape(xi)[1], 0)
            if context is not None:
                ci = context[:, start : start + subsize]
            else:
                ci = None
            if config.low_memory_mode and config.fusion_low_memory and train:
                out.append(
                    featurizer_w_recompute(
                        xi,
                        encoder,
                        config,
                        train,
                        reuse=reuse,
                        lengths=lengths,
                        max_length=subsize,
                        **kwargs
                    )
                )
            else:
                out.append(
                    featurizer(
                        xi,
                        encoder,
                        config,
                        train,
                        reuse=reuse,
                        lengths=lengths,
                        max_length=subsize,
                        context=ci,
                        **kwargs
                    )
                )
            reuse = True
        return merge_output_state(out, X, orig_lengths, config.chunk_pos_embed, hidden_dim=config.n_embed)

    return internal

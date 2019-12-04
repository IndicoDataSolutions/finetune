import tensorflow as tf

from finetune.util.shapes import shape_list

def linear(inp, output_dim, layer_name):
    with tf.variable_scope(layer_name):
        nx = shape_list(inp)[-1]
        if output_dim is None:
            output_dim = nx
        W = tf.get_variable(name="W", shape=[nx, output_dim], initializer=tf.initializers.glorot_normal())
        if inp.dtype == tf.float16:
            W = tf.cast(W, tf.float16)
            
        return tf.reshape(
            tf.matmul(
                tf.reshape(inp, [-1, nx]),
                tf.reshape(W, [nx, output_dim])
            ),
            shape_list(inp)[:-1] + [output_dim]
        )


def unfold(inp, kernel_size, padding="causal"):
    assert padding.lower() == "causal"
    slices = []
    paddings = [[0, 0], [kernel_size - 1, 0], [0, 0]]
    padded_input = tf.pad(inp, paddings, "CONSTANT")
    leng = tf.shape(padded_input)[1]
    for i in range(kernel_size):
        slices.append(padded_input[:, i: leng - kernel_size + i + 1])
    return tf.stack(slices, 2)


def dynamic_conv(inp, inp2=None, n_heads=8, kernel_size=4):
    batch, seq, n_channels = shape_list(inp)
    assert n_channels % n_heads == 0
    h = n_channels // n_heads
    
    unfolded = unfold(inp, kernel_size) # batch, length, kernel_size, channels 
    if inp2 is not None:
        # inp2: batch, length, ra_size, channels
        unfolded = tf.concat((unfolded, inp2), 2)
        kernel_size = shape_list(unfolded)[2]
        
    weights_linear = tf.reshape(
        linear(inp, n_heads * kernel_size, "kernel_machine",),
        [batch, seq, kernel_size, n_heads, 1], # batch time heads, kernel_size, 1
    )
    print(weights_linear, unfolded)
    unfolded = tf.reshape(unfolded, [batch, seq, kernel_size, n_heads, h])
    weights = tf.nn.softmax(weights_linear, -2)
    return tf.reshape(tf.reduce_sum(weights * unfolded, 2), shape_list(inp)) # batch, time heads, channels


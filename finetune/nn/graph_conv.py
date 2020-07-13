import tensorflow as tf


def _overlaps(values, low_key, high_key):
    low_values_x = tf.expand_dims(values[low_key], 1)  # batch, 1, seq
    high_values_x = tf.expand_dims(values[high_key], 1)  # batch, 1 seq

    low_values_y = tf.expand_dims(values[low_key], 2)  # batch, seq, 1
    high_values_y = tf.expand_dims(values[high_key], 2)  # batch, seq, 1
    start_contained = tf.logical_and(
        low_values_y <= low_values_x, low_values_x < high_values_y
    )
    end_contained = tf.logical_and(
        low_values_y < high_values_x, high_values_x <= high_values_y
    )
    return tf.cast(tf.logical_or(start_contained, end_contained), tf.float32)


def overlaps(values):
    return {
        "x": _overlaps(values, "left", "right"),
        "y": _overlaps(values, "top", "bottom"),
    }


def _direction(values, low_key, high_key):
    low_values_x = tf.expand_dims(values[low_key], 1)  # batch, 1, seq
    high_values_x = tf.expand_dims(values[high_key], 1)  # batch, 1 seq

    low_values_y = tf.expand_dims(values[low_key], 2)  # batch, seq, 1
    high_values_y = tf.expand_dims(values[high_key], 2)  # batch, seq, 1
    return {
        "before": tf.cast(
            high_values_y < low_values_x, tf.float32
        ),  # for dims (batch, in, out) this says in is above out,
        "after": tf.cast(
            low_values_y > high_values_x, tf.float32
        ),  # for dims (batch, in, out) this says in is above out,
    }


def directions(values):
    above_below = _direction(values, "top", "bottom")
    left_right = _direction(values, "left", "right")
    return {
        "above": above_below["before"],
        "below": above_below["after"],
        "left": left_right["before"],
        "right": left_right["after"],
    }


def graph_heads(values, sequence_lengths):
    overlap_matrix = overlaps(values)
    direction_matrix = directions(values)
    identity = tf.expand_dims(tf.eye(tf.shape(values["top"])[1]), 0)
    identity_mask = 1.0 - identity
    sequence_mask = tf.sequence_mask(
        sequence_lengths, dtype=tf.float32
    )  # 1 in areas we want to keep
    sequence_mask = tf.expand_dims(sequence_mask, 1) * tf.expand_dims(
        sequence_mask, -1
    )  # batch, seq, seq
    mask = identity_mask * sequence_mask
    return {
        "above": direction_matrix["above"] * overlap_matrix["x"] * mask,
        "below": direction_matrix["below"] * overlap_matrix["x"] * mask,
        "left": direction_matrix["left"] * overlap_matrix["y"] * mask,
        "right": direction_matrix["right"] * overlap_matrix["y"] * mask,
    }


class PairwiseDistance(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), int(input_shape[-1])]
        )

    def call(self, input):
        projected = tf.matmul(input, self.kernel)
        reshaped = tf.reshape(
            projected, tf.concat(tf.shape(projected)[:-1], [self.num_outputs, -1])
        )  # batch, seq, num_outputs, feats / num_outputs (new_feats)

        transposed = tf.transpose(
            a=reshaped, perm=[0, 2, 1, 3]
        )  # batch, num_out, seq, feats
        similarity = tf.matmul(
            transposed, transposed, transpose_b=True
        )  # batch, num_out, seq, seq
        return tf.transpose(similarity, [0, 2, 3, 1])  # batch, seq, seq, num


class GraphSmoothing(tf.keras.layers.Layer):
    def __init__(self, num_classes, num_convolutions=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_convolutions = num_convolutions

    def build(self, input_shape):
        self.final_gcn = GraphConvolution(self.num_classes)
        self.other_gcns = [
            GraphConvolution(int(input_shape[0][-1]))
            for _ in range(self.num_convolutions - 1)
        ]

    def call(self, inputs):
        features, positions, sequence_lengths = inputs
        graph = graph_heads(positions, sequence_lengths)
        adjacency_mat = tf.stack(
            [graph["above"], graph["below"], graph["left"], graph["right"]], 1
        )
        for gcn in self.other_gcns + [self.final_gcn]:
            features = gcn([features, adjacency_mat])

        return features


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        feature_shape, adjacency_mat_shape = input_shape
        feature_dim = int(feature_shape[-1])
        adjacency_heads = int(adjacency_mat_shape[1])
        self.kernel = self.add_weight(
            "kernel", shape=[adjacency_heads, feature_dim, self.num_outputs]
        )
        self.bias = self.add_weight("kernel", shape=[adjacency_heads, self.num_outputs, 1])

    def call(self, inputs):
        features, adjacency_mat = inputs
        features_with_head_dim = tf.expand_dims(features, 1)  # batch, 1, seq, feats
        # hopefully this will broadcast across heads dim?
        projected = tf.matmul(features_with_head_dim, self.kernel)
        # projected = batch, heads, seq, n_out
        # adjacency_mat = batch, heads, seq, seq
        adjacency_mat = tf.nn.relu(adjacency_mat)  # remove any spurious negative values
        normed_adjacency_mat = tf.math.divide_no_nan(
            adjacency_mat, tf.reduce_sum(adjacency_mat, 2, keepdims=True)
        )
        shared = (
            tf.matmul(projected, normed_adjacency_mat, transpose_a=True) + self.bias
        )  # batch, n_out, heads, seq
        return tf.transpose(tf.reduce_mean(shared, 1), [0, 2, 1])  # batch, seq, n_out

import tensorflow as tf

from finetune.util.timing import ProgressBar


class InputMode:
    PREDICT = "predict"
    TRAIN = "train"


def has_targets(generator):
    sample = next(iter(generator()))
    return isinstance(sample, tuple) and len(sample) == 2


def add_length(x, y=None):
    x["length"] = tf.shape(x["tokens"])[0]
    if y is not None:
        return x, y
    return x


def batch_dataset(
    dataset: tf.data.Dataset,
    batch_size: int,
    shapes: dict[str, tf.TensorShape]
    | tuple[dict[str, tf.TensorShape], dict[str, tf.TensorShape]],
    max_length: int,
    n_epochs: int = 1,
    shuffle: bool = False,
    table_batching: bool = False,
    random_seed: int = 42,
):
    if isinstance(shapes, tuple):
        shapes = ({**shapes[0], "length": tf.TensorShape([])}, shapes[1])
    else:
        shapes = {**shapes, "length": tf.TensorShape([])}

    if table_batching:
        assert isinstance(
            shapes, tuple
        ), "You cannot use table batching to predict on tables as order is not guarenteed"

        return (
            dataset.map(add_length)
            .shuffle(500 if shuffle else 1, seed=random_seed)
            # When we update to tf.2.13 this will change to be a method on the dataset.
            .apply(
                tf.data.experimental.bucket_by_sequence_length(
                    element_length_func=(
                        lambda item, *_: tf.cast(
                            tf.maximum(
                                tf.reduce_sum(
                                    item["context"][:, 0] - item["context"][:, 2] + 1
                                ),
                                tf.reduce_sum(
                                    item["context"][:, 1] - item["context"][:, 3] + 1
                                ),
                            ),
                            tf.int32,
                        )
                    ),
                    bucket_boundaries=[max_length],
                    bucket_batch_sizes=[batch_size, 1],
                    padded_shapes=shapes,
                    drop_remainder=False,
                )
            )
            .repeat(n_epochs)
            .prefetch(tf.data.AUTOTUNE)
        )

    else:
        return (
            dataset.map(add_length)
            .shuffle(500 if shuffle else 1, seed=random_seed)
            .padded_batch(batch_size, padded_shapes=shapes, drop_remainder=False)
            .repeat(n_epochs)
            .prefetch(tf.data.AUTOTUNE)
        )


def wrap_tqdm(
    gen,
    mode,
    n_epochs,
    dataset_size,
    current_epoch_offset=0,
    total_epoch_offset=0,
    quiet=False,
    update_hook=None,
):
    assert mode in {"train", "predict"}
    if mode == "predict":
        return gen  # tqdm is handled elsewhere (not sure why)

    try:
        total = len(gen)
    except:
        total = dataset_size
    epoch = 1

    def internal_gen():
        nonlocal epoch
        current_epoch = (epoch - 1) % n_epochs + 1
        it = iter(gen())
        desc = "Epoch {}/{}".format(
            current_epoch + current_epoch_offset, n_epochs + total_epoch_offset
        )
        for i in ProgressBar(
            it,
            desc=desc,
            total=total,
            miniters=1,
            leave=current_epoch == n_epochs and mode == "train",
            update_hook=update_hook,
            quiet=quiet,
            current_epoch=current_epoch + current_epoch_offset,
            total_epochs=n_epochs + total_epoch_offset,
        ):
            yield i
        if mode == "train":
            epoch += 1

    return internal_gen


class Chunker:
    def __init__(self, max_length, total_context_width, justify="c"):
        if total_context_width is None:
            total_context_width = 2 * max_length // 3
        assert total_context_width < max_length
        assert justify.lower() in {"center", "left", "right"}

        self.max_length = max_length
        self.total_context_width = total_context_width
        self.chunk_size = self.max_length - 2
        self.useful_chunk_width = self.chunk_size - total_context_width
        self.justify = justify.lower()

        if self.justify == "left":
            self.normal_start = 0
        elif self.justify == "right":
            self.normal_start = total_context_width
        elif self.justify == "center":
            self.normal_start = total_context_width // 2

        self.normal_end = self.normal_start + self.useful_chunk_width

    def generate_chunks(self, length):
        for start in range(0, length, self.useful_chunk_width):
            end = start + self.chunk_size
            is_start = start == 0
            is_end = end >= length
            yield start, end, self.useful_chunk_section(is_start, is_end)
            if is_end:
                break

    def useful_chunk_section(self, start_of_doc, end_of_doc):
        start = self.normal_start
        end = self.normal_end
        if start_of_doc:
            start = 0
        if end_of_doc:
            end = self.max_length
        return start, end

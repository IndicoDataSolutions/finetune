import math

import tensorflow as tf

from finetune.util.timing import ProgressBar


class InputMode:
    PREDICT = "predict"
    TRAIN = "train"


def _integer_val_size(val_size, dataset_size):
    if isinstance(val_size, float):
        return int(val_size * dataset_size)
    return val_size


def validation_settings(
    dataset_size, batch_size, val_size, val_interval, keep_best_model
):
    """
    Auto-select reasonable validation settings
    """
    if val_size is not None and val_interval is not None:
        return (
            _integer_val_size(val_size, dataset_size),
            val_interval,
        )

        # Auto-select reasonable validation size
    if val_size == "auto":
        if dataset_size < 50 and not keep_best_model:
            val_size = 0
        else:
            val_size = max(5, int(0.05 * dataset_size))
            val_size = min(100, val_size)
    else:
        val_size = _integer_val_size(val_size, dataset_size)

        # Auto-select reasonable validation interval
    if val_interval is None:
        # sys.maxsize corresponds to never running validation
        # and is used when val_size is set to 0
        val_interval = 4 * int(math.ceil(val_size / batch_size)) or None
    else:
        val_interval = int(val_interval)

    return int(val_size), val_interval


def has_targets(generator):
    sample = next(iter(generator()))
    return isinstance(sample, tuple) and len(sample) == 2


def add_length(x, y=None):
    x["length"] = tf.shape(x["tokens"])[0]
    if y is not None:
        return x, y
    return x


def batch_dataset(dataset, batch_size, shapes, n_epochs=1):
    if isinstance(shapes, tuple):
        shapes = ({**shapes[0], "length": tf.TensorShape([])}, shapes[1])
    else:
        shapes = {**shapes, "length": tf.TensorShape([])}

    def batched_dataset():
        return (
            dataset()
            .map(add_length)
            .padded_batch(batch_size, padded_shapes=shapes, drop_remainder=False)
            .repeat(n_epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    return batched_dataset


def wrap_tqdm(
    gen,
    mode,
    n_epochs,
    val_size,
    dataset_size,
    current_epoch_offset=0,
    total_epoch_offset=0,
    skip_val=False,
    quiet=False,
    update_hook=None,
):
    assert mode in {"train", "predict", "evaluate"}
    if mode == "predict":
        return gen  # tqdm is handled elsewhere (not sure why)

    try:
        total = len(gen)
    except:
        if mode == "train":
            total = dataset_size
        else:
            total = val_size

    epoch = 1

    def internal_gen():
        nonlocal epoch
        current_epoch = (epoch - 1) % n_epochs + 1
        it = iter(gen())
        if mode == "train":
            desc = "Epoch {}/{}".format(
                current_epoch + current_epoch_offset, n_epochs + total_epoch_offset
            )
        else:
            desc = "Validation"
        if skip_val:
            for _, i in zip(range(val_size), it):
                yield i
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

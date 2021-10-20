import time
from tqdm import tqdm


class ProgressBar(tqdm):
    """Provides a `total_time` format parameter"""

    def __init__(self, *args, **kwargs):
        self._update_hook = kwargs.pop("update_hook", None)
        self._quiet = kwargs.pop("quiet", False)
        self.current_epoch = kwargs.pop("current_epoch", 1)
        self.total_epochs = kwargs.pop("total_epochs", 1)
        self.remaining_epochs = self.total_epochs - self.current_epoch
        self._quiet_update_frequency = kwargs.pop("_quiet_update_frequency", 30)
        self.last_update = 0
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        remaining_iterations = d["total"] - d["n"]
        remaining_seconds_this_epoch = remaining_iterations / (d.get("rate") or 1.0)
        d["estimated_time_per_epoch"] = remaining_seconds_this_epoch + d["elapsed"]
        d["estimated_total_seconds"] = d["estimated_time_per_epoch"] * self.total_epochs
        d["estimated_remaining_seconds"] = (
            remaining_seconds_this_epoch
            + self.remaining_epochs * d["estimated_time_per_epoch"]
        )
        return d

    def display(self, *args, **kwargs):
        """
        Be careful to avoid expensive update hooks
        """
        curr_time = time.time()
        update = False
        if curr_time - self.last_update > self._quiet_update_frequency:
            update = True
            self.last_update = curr_time

        if self._update_hook is not None and update:
            self._update_hook(self.format_dict)
        if self._quiet:
            if update:
                self.fp.write(
                    "{prefix}: {n} / {total} {unit} Estimated {estimated_remaining_seconds} seconds remaining. \n".format(
                        **self.format_dict
                    )
                )
                self.fp.flush()
        else:
            return super().display(*args, **kwargs)

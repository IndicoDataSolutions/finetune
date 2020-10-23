from tqdm import tqdm


class ProgressBar(tqdm):
    """Provides a `total_time` format parameter"""

    def __init__(self, *args, **kwargs):
        self._update_hook = kwargs.pop("update_hook", None)
        self._silent = kwargs.pop("silent", False)
        self.current_epoch = kwargs.pop("current_epoch", 1)
        self.total_epochs = kwargs.pop("total_epochs", 1)
        self.remaining_epochs = self.total_epochs - self.current_epoch
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
        if self._update_hook is not None:
            self._update_hook(self.format_dict)

        if not self._silent:
            return super().display(*args, **kwargs)

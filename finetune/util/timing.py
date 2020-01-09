from tqdm import tqdm


class ProgressBar(tqdm):
    """Provides a `total_time` format parameter"""

    def __init__(self, *args, **kwargs):
        self._update_hook = kwargs.pop('update_hook', None)
        self._silent = kwargs.pop("silent", False)
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        remaining_iterations = d['total'] - d['n']
        d['estimated_remaining_seconds'] = remaining_iterations / (d.get('rate') or 1.)
        d['estimated_total_seconds'] = d['estimated_remaining_seconds'] + d['elapsed'] 
        return d
        
    def display(self, *args, **kwargs):
        """
        Be careful to avoid expensive update hooks
        """
        if self._update_hook is not None:
            self._update_hook(self.format_dict)

        if not self._silent:
            return super().display(*args, **kwargs)

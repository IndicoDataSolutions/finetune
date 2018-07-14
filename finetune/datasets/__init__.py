import pandas as pd

class Dataset:

    def __init__(self, filename=None):
        self.filename = filename
        self.download()

    def download(self):
        raise NotImplementedError

    def dataframe(self, nrows=None):
        return pd.read_csv(self.filename, nrows=nrows)

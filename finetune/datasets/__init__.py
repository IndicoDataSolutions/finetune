import pandas as pd

class Dataset:

    def __init__(self, filename=None, nrows=None):
        self.filename = filename
        self.download()
        self.dataframe = pd.read_csv(self.filename, nrows=nrows)

    def download(self):
        raise NotImplementedError

import hashlib
from io import StringIO
import os.path
from pathlib import Path
import requests
import pandas as pd


def file_hash(path_obj):
    h = hashlib.sha256()
    with path_obj.open(mode='rb', buffering=0) as f:
        for b in iter(lambda: f.read(128 * 1024), b''):
            h.update(b)
    return str(h.hexdigest())


class Dataset:
    def __init__(self, filename=None, nrows=None):
        self.filename = filename
        self.maybe_download()
        self.dataframe = pd.read_csv(self.filename, nrows=nrows).dropna()

    @property
    def md5(self):
        raise NotImplementedError

    def maybe_download(self):
        path = Path(self.filename)
        if path.exists() and file_hash(path) == self.md5:
            return
        else:
            self.download()

    def download(self):
        raise NotImplementedError


def generic_download(url, text_column, target_column, filename, save=True, task_type='Classify', data_directory='Data', text_transformation=None, target_transformation=None):

    save_path = os.path.join(data_directory, task_type, filename)
    if os.path.exists(save_path):
        print("{} already downloaded, skipping...".format(filename))
        return

    response = requests.get(url)
    _file = StringIO(response.text.replace('\r', '\n'))
    df = pd.read_csv(_file)
    df = df.dropna(subset=[text_column, target_column])

    new_df = pd.DataFrame(columns=['Text', 'Target'])
    new_df['Text'], new_df['Target'] = df[text_column], df[target_column]

    if text_transformation is not None:
        new_df['Text'] = new_df['Text'].apply(text_transformation)
    if target_transformation is not None:
        new_df['Target'] = new_df['Target'].apply(target_transformation)

    if save:
        new_df.to_csv(save_path, index=False)

    return new_df


def comparison_download(url, text_column1, text_column2, target_column, filename, save=True, task_type='Similarity', data_directory='Data'):

    save_path = os.path.join(data_directory, task_type, filename)

    response = requests.get(url)
    _file = StringIO(response.text.replace('\r', '\n'))
    df = pd.read_csv(_file, sep="\t")
    df = df.dropna(subset=[text_column1, text_column2, target_column])

    new_df = pd.DataFrame(columns=['Text1', 'Text2', 'Target'])
    new_df['Text1'], new_df['Text2'], new_df['Target'] = df[text_column1], df[text_column2], df[target_column]

    if save:
        new_df.to_csv(save_path, index=False)

    return new_df


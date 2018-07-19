from io import StringIO
import os.path

import requests
import pandas as pd


class Dataset:

    def __init__(self, filename=None, nrows=None):
        self.filename = filename
        self.download()
        self.dataframe = pd.read_csv(self.filename, nrows=nrows).dropna()

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

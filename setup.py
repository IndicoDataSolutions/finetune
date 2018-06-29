"""
IndicoAPI setup
"""
import os
from sys import version_info
from setuptools import setup, find_packages
from pathlib import Path

REQUIREMENTS = [
    "pandas>=0.23.1",
    "IndicoIo>=1.1.5",
    "enso>=0.1.6",
    "tqdm>=4.0.0",
    "numpy>=1.13.0",
    "scikit-learn>=0.18.0",
    "joblib>=0.11",
    "ftfy>=4.4.0",
    "spacy>=2.0.0",
    "msgpack-numpy>=0.4.1",
    "sh>=1.12.14"
]

setup(
    name="finetune",
    packages=find_packages(),
    version="0.1.0",
    install_requires=REQUIREMENTS,
    include_package_data=False
)

# Download data
from tqdm import tqdm
import requests
import finetune
from finetune.model import SHAPES_PATH

def download_data():
    if Path(SHAPES_PATH).exists():
        # files already downloaded
        print("Data files already present")
        return

    base_url = "https://raw.githubusercontent.com/IndicoDataSolutions/finetune/master/model/"

    file_list = [
        "encoder_bpe_40000.json",
        "params_0.npy",
        "params_1.npy",
        "params_2.npy",
        "params_3.npy",
        "params_4.npy",
        "params_5.npy",
        "params_6.npy",
        "params_7.npy",
        "params_8.npy",
        "params_9.npy",
        "param_shapes.json",
        "vocab_40000.bpe",
    ]

    print("Downloading files...")
    for filename in tqdm(file_list):
        data = requests.get(base_url + filename).content
        local_filepath = os.path.join(
            os.path.dirname(finetune.__file__),
            'model',
            filename
        )
        fd = open(local_filepath, 'wb')
        fd.write(data)
        fd.close()


download_data()

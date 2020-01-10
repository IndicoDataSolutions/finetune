"""
IndicoAPI setup
"""
import os
from sys import version_info
from setuptools import setup, find_packages


REQUIREMENTS = [
    "pandas>=0.23.1",
    "tqdm>=4.0.0",
    "numpy>=1.13.0",
    "scipy>=1.1.0",
    "scikit-learn>=0.20.2,<=0.22",
    "ftfy>=4.4.0",
    "spacy>=2.0.0",
    "h5py>=2.8.0",
    "joblib>=0.12.0",
    "bs4>=0.0.1",
    "imbalanced-learn>=0.6.0,<0.7.0",
    "nltk>=3.2.4",
    "regex>=2019.03.12",
    "lxml>=4.3.3"
    "sentencepiece>=0.1.83"
]

setup(
    name="finetune",
    packages=find_packages(),
    version="0.8.6",
    install_requires=REQUIREMENTS,
    extras_require={
        "tf": ["tensorflow==1.14.0"],
        "tf_gpu": ["tensorflow-gpu==1.14.0"],
    },
    include_package_data=False
)

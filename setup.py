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
    "tqdm>=4.0.0",
    "numpy>=1.13.0",
    "scipy>=1.1.0",
    "scikit-learn>=0.18.0",
    "ftfy>=4.4.0",
    "spacy>=2.0.0",
    "msgpack-numpy==0.4.1",
    "h5py>=2.8.0",
    "joblib>=0.12.0",
    "bs4>=0.0.1",
    "imblearn>=0.0"
]

setup(
    name="finetune",
    packages=find_packages(),
    version="0.5.1",
    install_requires=REQUIREMENTS,
    include_package_data=False
)

print("\nDownloading required model files...")
from finetune.download import download_data_if_required
download_data_if_required()

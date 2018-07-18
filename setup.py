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
    "ftfy>=4.4.0",
    "spacy>=2.0.0",
    "msgpack-numpy>=0.4.1",
    "h5py>=2.8.0",
    "tensorflow-gpu>=1.6"
]

setup(
    name="finetune",
    packages=find_packages(),
    version="0.1.0",
    install_requires=REQUIREMENTS,
    include_package_data=False
)

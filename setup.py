"""
IndicoAPI setup
"""
from sys import version_info
from setuptools import setup, find_packages

REQUIREMENTS = [
    "tqdm>=4.0.0",
    "pandas>=0.20.0",
    "numpy>=1.13.0",
    "scikit-learn>=0.18.0",
    "joblib>=0.11",
    "ftfy>=4.4.0",
    "spacy>=2.0.0",
    "enso>=0.1.1",
    "tensorflow-gpu>=1.6.0",
]

setup(
    name="finetune",
    packages=find_packages(),
    version="0.1.0",
    install_requires=REQUIREMENTS,
)

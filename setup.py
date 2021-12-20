"""
IndicoAPI setup
"""
import os
import warnings
import subprocess
from sys import version_info, argv
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

REQUIREMENTS = [
    "pandas>=0.23.1",
    "tqdm>=4.0.0",
    "numpy>=1.18.4",
    "scipy>=1.1.0",
    "scikit-learn>=0.20.2,<0.23",
    "ftfy>=4.4.0",
    "spacy>=2.0.0,<3.0.0",
    "h5py>=2.8.0",
    "joblib>=0.12.0",
    "bs4>=0.0.1",
    "imbalanced-learn>=0.6.0,<0.7.0",
    "nltk>=3.2.4",
    "regex>=2019.03.12",
    "lxml>=4.3.3",
    "sentencepiece>=0.1.83",
    "tabulate>=0.8.6,<0.9.0",
    "tensorflow-addons==0.11.2",
    "tensorflow-estimator",
    "tqdl==0.0.4",
    "psutil==5.7.0",
    "transformers==4.5.1",
]


class OpsBuild(build_ext):
    def run(self):
        script = os.path.join(
            os.path.dirname(__file__), "finetune", "custom_ops", "build.sh"
        )
        if subprocess.run(["sh", script]).returncode != 0:
            warnings.warn(
                "Failed to build the finetune memory management ops required for use of Scheduler. "
                "If you don't intend to use Scheduler you can safely ignore this message. "
                "To build the ops later execute {}".format(script)
            )


setup(
    name="finetune",
    packages=find_packages(),
    version="0.9.0",
    install_requires=REQUIREMENTS,
    extras_require={
        "tf": ["tensorflow==2.3.0"],
        "tf_gpu": ["tensorflow-gpu==2.3.0"],
    },
    zip_safe=False,
    cmdclass={"build_ext": OpsBuild,},
    include_package_data=True
)

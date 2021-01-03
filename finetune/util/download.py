import os
import urllib
import urllib.request
from pathlib import Path
import tqdl

import finetune

FINETUNE_BASE_FOLDER = os.path.dirname(finetune.__file__)
GPT_BASE_URL     = "https://s3.amazonaws.com/bendropbox/gpt/"
GPT2_BASE_URL    = "https://s3.amazonaws.com/bendropbox/gpt2/"
BERT_BASE_URL    = "https://s3.amazonaws.com/bendropbox/bert/"
ROBERTA_BASE_URL = "https://s3.amazonaws.com/bendropbox/roberta/"
OSCAR_BASE_URL   = "https://s3.amazonaws.com/bendropbox/oscar/"
LAYOUTLM_BASE_URL   = "https://s3.amazonaws.com/bendropbox/layoutlm/"

def download_data_if_required(base_model):
    """ Pulls the pre-trained model weights from Github if required. """

    for file_obj in base_model.required_files:
        path = Path(file_obj['file'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading: {}".format(path.name))
            tqdl.download(file_obj['url'], str(path))

    base_model.translate_base_model_format()

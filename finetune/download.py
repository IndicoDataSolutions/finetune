import os
import urllib
import urllib.request
from urllib.parse import urljoin
from pathlib import Path

import finetune


def download_data_if_required():
    """ Pulls the pre-trained model weights from Github if required. """
    github_base_url = "https://raw.githubusercontent.com/IndicoDataSolutions/finetune/master/finetune/model/"
    s3_base_url = "https://s3.amazonaws.com/bendropbox/"
    gpt2_base_url = "https://s3.amazonaws.com/bendropbox/gpt2/"
    finetune_base_folder = os.path.dirname(finetune.__file__)
    file_list = [
        {
            'file': os.path.join(finetune_base_folder, 'model', 'gpt', 'encoder.json'),
            'url': urljoin(github_base_url, "encoder_bpe_40000.json")
        },
        {
            'file': os.path.join(finetune_base_folder, 'model', 'gpt', 'vocab.bpe'),
            'url': urljoin(github_base_url, "vocab_40000.bpe")
        },
        {
            'file': os.path.join(finetune_base_folder, 'model', 'gpt', 'model-lg.jl'),
            'url': urljoin(s3_base_url, "Base_model.jl")
        },
        {
            'file': os.path.join(finetune_base_folder, 'model', 'gpt', 'model-sm.jl'),
            'url': urljoin(s3_base_url, "SmallBaseModel.jl")
        },
        {
            'file': os.path.join(finetune_base_folder, 'model', 'gpt2', 'encoder.json'),
            'url': urljoin(gpt2_base_url, 'encoder.json')
        },
        {
            'file': os.path.join(finetune_base_folder, 'model', 'gpt2', 'vocab.bpe'),
            'url': urljoin(gpt2_base_url, 'vocab.bpe')
        },
        {
            'file': os.path.join(finetune_base_folder, 'model', 'gpt2', 'model-sm.jl'),
            'url': urljoin(gpt2_base_url, 'model-sm.jl')
        }
    ]

    for file_obj in file_list:
        path = Path(file_obj['file'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading: {}".format(path.name))
            data = urllib.request.urlopen(file_obj['url']).read()
            with path.open('wb') as f:
                f.write(data)

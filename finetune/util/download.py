import os
import urllib
import urllib.request
from pathlib import Path

import finetune


def download_data_if_required():
    """ Pulls the pre-trained model weights from Github if required. """
    github_base_url = "https://raw.githubusercontent.com/IndicoDataSolutions/finetune/master/finetune/model/"
    s3_base_url = "https://s3.amazonaws.com/bendropbox/"

    file_list = [
        (github_base_url, "encoder_bpe_40000.json"),
        (github_base_url, "vocab_40000.bpe"),
        (s3_base_url, "Base_model.jl"),
        (s3_base_url, "SmallBaseModel.jl")
    ]

    for root_url, filename in file_list:
        folder = os.path.abspath(os.path.join(
            os.path.dirname(finetune.__file__), 'model'
        ))
        if not os.path.exists(folder):
            os.mkdir(folder)

        local_filepath = os.path.join(folder, filename)

        if not Path(local_filepath).exists():
            print("Downloading: {}".format(local_filepath))
            data = urllib.request.urlopen(root_url + filename).read()
            fd = open(local_filepath, 'wb')
            fd.write(data)
            fd.close()

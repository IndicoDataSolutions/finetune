# Download data
import os
from pathlib import Path

import requests

import finetune


def download_data_if_required():
    """ Pulls the pre-trained model weights from Github if required. """
    base_url = "https://raw.githubusercontent.com/IndicoDataSolutions/finetune/master/finetune/model/"

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
        "params_shapes.json",
        "vocab_40000.bpe",
    ]

    for filename in file_list:
        folder = os.path.join(
            os.path.dirname(finetune.__file__),
            'model'
        )
        if not os.path.exists(folder):
            os.mkdir(folder)

        local_filepath = os.path.join(folder, filename)

        if not Path(local_filepath).exists():
            data = requests.get(base_url + filename).content
            fd = open(local_filepath, 'wb')
            fd.write(data)
            fd.close()


if __name__ == "__main__":
    download_data_if_required()

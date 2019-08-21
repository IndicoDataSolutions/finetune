Installation
============
Finetune can be installed directly from PyPI by using `pip`

.. code-block:: bash

    pip install finetune


or installed directly from source:

.. code-block:: bash

    git clone https://github.com/IndicoDataSolutions/finetune
    cd finetune
    python3 setup.py develop
    python3 -m spacy download en

You can optionally run the provided test suite to ensure installation completed successfully.

.. code-block:: bash

    pip3 install pytest
    pytest

Docker
======

If you'd prefer you can also run :mod:`finetune` in a docker container. The bash scripts provided assume you have a functional install of `docker <https://docs.docker.com/install>`_ and `nvidia-docker <https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>`_.

.. code-block:: bash

    # For usage with NVIDIA GPUs
    ./docker/build_gpu_docker.sh  # builds a docker image
    ./docker/start_gpu_docker.sh  # starts a docker container in the background
    docker exec -it finetune bash # starts a bash session in the docker container

For CPU-only usage:

.. code-block:: bash

    ./docker/build_cpu_docker.sh
    ./docker/start_cpu_docker.sh

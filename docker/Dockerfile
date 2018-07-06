FROM tensorflow/tensorflow:1.6.0-gpu-py3
LABEL Author="Ben Townsend, Madison May"

# tensorboard
EXPOSE 6006

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=8.0"

RUN apt-get update
RUN mkdir /Finetune
ADD requirements.txt /Finetune/requirements.txt
RUN pip3 install -r /Finetune/requirements.txt
RUN python3 -m spacy download en
ENV LANG C.UTF-8
WORKDIR /Finetune
CMD ["sleep","infinity"]

FROM python:3.13.8-slim

RUN pip3 install tensorflow[and-cuda]==2.20.0
RUN apt update && apt install -y cmake g++

# tensorboard
EXPOSE 6006
RUN apt-get update && mkdir /Finetune
WORKDIR /Finetune

CMD ["sleep", "infinity"]

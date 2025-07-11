FROM python:3.12-slim

RUN pip3 install tensorflow[and-cuda]==2.19.0
RUN apt update && apt install -y cmake g++ 

# tensorboard
EXPOSE 6006
RUN apt-get update && mkdir /Finetune
WORKDIR /Finetune

CMD ["sleep", "infinity"]

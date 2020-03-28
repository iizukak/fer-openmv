FROM tensorflow/tensorflow:2.0.0-gpu AS base 

LABEL maintainer="iizuka.kentaro@gmail.com"

# For TensorBoard
EXPOSE 6006

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-pil

WORKDIR /home/fer
COPY requirements.txt /home/fer/
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt

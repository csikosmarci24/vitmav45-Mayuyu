FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update

COPY project/* project/
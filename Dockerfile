FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# https://github.com/NVIDIA/nvidia-docker/issues/1632
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
RUN pip install gdown
RUN pip install pandas
RUN pip install networkx

COPY project/ project/

RUN cd project/data &&\
    mkdir files &&\
    cd files &&\
    gdown 1Ot-ICpiJRlisFvM9Fi6TM3Q6kAZaLS0y &&\
    gdown 1LSdAthCa69kWRIKoI5UmclLgf4OsSNAm &&\
    gdown 15_hqow9NT_M49OX7cXrG5P6vCgfbKyhP

RUN export PYTHONPATH="${PYTHONPATH}:."

EXPOSE 4000

# Commented for testing
# CMD ["python", "project/hello.py"]
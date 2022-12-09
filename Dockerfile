FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN apt-get update

COPY project/ project/
COPY environment.yml .

RUN conda env create -f environment.yml

EXPOSE 4000

ENTRYPOINT [ "conda", "run", "-n", "mayuyu_clone", "--live-stream", "python", "project/train.py" ]

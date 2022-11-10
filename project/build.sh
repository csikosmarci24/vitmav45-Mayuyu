#!/bin/sh
docker build . -t mayuyu
docker run --gpus all -it mayuyu

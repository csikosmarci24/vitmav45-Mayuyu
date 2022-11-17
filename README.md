# Team information

Team name: Mayuyu

Members:

| Name | NEPTUN |
| ---- | ------ |
| Balázs Tibor Morvay | CLT8ZP |
| László Kálmán Trautsch | CMLJKQ |
| Marcell Csikós | P78P08 |

# Project information

Selected topic: Identification of drug interactions with graph autoencoder

Our task is identifying if there are adverse reactions if a combination of drugs is used by a patient. For this task we will explore Variational Graph Auto-Encoder (VGAE) solutions. The model's input is an interaction graph showing which drugs have interference with each other. There is also the DrugBank database which contains more information on the individual drugs.

Our first step is using only the interaction graph without any features to rebuild the edges. After that the accuracy can be improved by using DrugBank features.

# Milestone 1

Data:
* DrugBank database (https://go.drugbank.com/)
* DrugBank interaction graph (http://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html)

Related files:
* project/data_visualization.ipynb: Notebook with the download scripts, visualization and processing of the data
* project/100edges.html: Additional visualization of the interaction graph with its first 100 edges

# Running the code

Build the image:

```
docker build . -t <image-name>
```

Run container:

```
docker run --gpus all -it <image-name>
```

Test GPU access in container:
```
python project/hello.py
```

All steps with shell script (Linux):
```
chmod +x build.sh
./build.sh
```

# VSCode development in container

1. Install Remote Development extension
2. Command Palette (Ctrl+Shift+P): Dev Containers: Rebuild and Reopen in Container
3. Without Nvidia GPU: in .devcontainer/devcontainer.json comment "runArgs"

# Materials for team members

https://paperswithcode.com/paper/variational-graph-auto-encoders

https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

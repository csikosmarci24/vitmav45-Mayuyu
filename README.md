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

# Milestone 1

Data:
* DrugBank database (https://go.drugbank.com/)
* DrugBank interaction graph (http://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html)

Related files:
* project/data_visualization.ipynb

# Running the code (for later stages)

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

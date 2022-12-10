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
* project/data_visualization.ipynb: Notebook with the download scripts, visualization and processing of the data.
* project/100edges.html: Additional visualization of the interaction graph with its first 100 edges.

# Milestone 2

Variational Graph Auto-Encoder model created. Training and evaluation works with and without extracted DrugBank features.

The model is based on: https://github.com/dmlc/dgl/tree/master/examples/pytorch/vgae

Related files:
* project/data/process_graph.py: Load interaction graph from file and return Networkx representation.
* project/data/process_drugbank.py: Load DrugBank database from file and return processed Pandas DataFrame.
* project/data/input_data.py: Return the interaction graph's adjacency matrix and the features from the database.
* project/model/preprocess.py: Splitting the graph to train, validation and test subgraphs for training.
* project/model/model.py: Implementation of the model.
* project/train.py: Training and evaluating the model.

Default hyperparameters (can be changed with command line arguments):
* Learning rate (--learning_rate): 0.01
* Number of epochs (-e): 300
* 1st hidden layer units (-h1): 32
* 2nd hidden layer units (-h2): 16

Wiki page on DrugBank features used during training: https://github.com/csikosmarci24/vitmav45-Mayuyu/wiki/Features-used-during-training

# Running the project with Conda

Create environment:
```
conda env create -f environment.yml
```

Activate environment:
```
conda activate mayuyu
```

Train and evaluate the model:
```
python project/train.py
```

For systems without CUDA-enabled GPUs, environment-cpu.yml can be used, with the environment name of mayuyu_cpu.

# Running the project with Docker (experimental)

Build the image:

```
docker build . -t <image-name>
```

Create and run container (starts training):

```
docker run --gpus all -it <image-name>
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

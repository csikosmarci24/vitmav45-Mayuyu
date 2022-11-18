import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import gzip
import zipfile
import pandas as pd
from project.data.process_drugbank import parse_drugbank_xml
from project.data.process_graph import parse_graph, create_network_graph_from

"""
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset)
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1
        )
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
"""


def load_data():
    nx_graph = create_network_graph_from(parse_graph())
    adj = nx.adjacency_matrix(nx_graph)

    df = parse_drugbank_xml()
    feature_df = df[['type', 'groups', 'ATC1', 'ATC2', 'ATC3', 'ATC4', 'ATC5']].copy()

    #features = sp.identity(adj.shape[0])
    features = sp.csr_matrix(feature_df.values)

    return adj, features

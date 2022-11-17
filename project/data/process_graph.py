import gzip
import pandas as pd
import networkx as nx
from pyvis import network as net

def get_train_val_test_split():
    """Returns the train, validation and test dataframes."""
    return split(parse_graph())

def parse_graph():
    """Parses the graph data and returns a pandas dataframe containing it."""
    with gzip.open('ChCh-Miner_durgbank-chem-chem.tsv.gz') as f:
        graph_df = pd.read_csv(f, delimiter='\t', names=['Source', 'Target'])
        return graph_df

def create_network_graph_from(data_frame):
    nx_graph = nx.from_pandas_edgelist(data_frame, source='Source', target='Target')
    return nx_graph

def create_graph_visualization_from(graph_df):
    graph_part = nx.from_pandas_edgelist(graph_df[0:10], source='Source', target='Target')
    g = net.Network(notebook=True)
    g.from_nx(graph_part)
    nx.draw(graph_part, nx.spring_layout(graph_part, k=0.75, iterations=20), with_labels=True)

def split(graph_df):
    """Splits the dataframe passed as parameter into train, val, and test dataframes, then returns those."""
    tenth_len = graph_df.shape[0] // 10
    train_edges = graph_df[0: (8 * tenth_len)]
    val_edges = graph_df[(8 * tenth_len): (9 * tenth_len)]
    test_edges = graph_df[(9 * tenth_len): -1]

    return train_edges, val_edges, test_edges

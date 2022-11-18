import gzip
import pandas as pd
import networkx as nx


def parse_graph():
    """Parses the graph data and returns a pandas dataframe containing it."""
    with gzip.open('data/ChCh-Miner_durgbank-chem-chem.tsv.gz') as f:
        graph_df = pd.read_csv(f, delimiter='\t', names=['Source', 'Target'])
        return graph_df


def create_network_graph_from(data_frame):
    nx_graph = nx.from_pandas_edgelist(data_frame, source='Source', target='Target')
    return nx_graph


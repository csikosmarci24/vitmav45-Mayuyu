import gzip
import pandas as pd
import networkx as nx
import os


def parse_graph():
    """Parses the graph data and returns a networkx graph created from it."""
    cwd = os.path.dirname(os.path.abspath(__file__))

    with gzip.open(os.path.join(cwd, 'files/ChCh-Miner_durgbank-chem-chem.tsv.gz')) as f:
        graph_df = pd.read_csv(f, delimiter='\t', names=['Source', 'Target'])
        nx_graph = nx.from_pandas_edgelist(graph_df, source='Source', target='Target')
        return nx_graph

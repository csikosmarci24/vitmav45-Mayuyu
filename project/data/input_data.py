import networkx as nx
import scipy.sparse as sp
from data.process_drugbank import parse_drugbank_xml
from data.process_graph import parse_graph
from data.fetch_data import fetch_data


def load_data():
    print('Checking and downloading files...')
    fetch_data()
    print('Processing interaction graph...')
    nx_graph = parse_graph()
    print('Processing DrugBank...')
    df = parse_drugbank_xml()

    # If drug is not in drugbank, then delete it from the graph
    del_list = []
    for id in list(nx_graph.nodes()):
        if id not in df.index:
            del_list.append(id)
    nx_graph.remove_nodes_from(del_list)

    adj = nx.adjacency_matrix(nx_graph)

    df = df.loc[list(nx_graph.nodes())]
    feature_df = df[['type', 'groups', 'ATC1', 'ATC2', 'ATC3', 'ATC4', 'ATC5']].copy()
    features = sp.csr_matrix(feature_df.values)

    return adj, features

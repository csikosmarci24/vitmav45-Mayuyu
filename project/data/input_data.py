import networkx as nx
import scipy.sparse as sp
import pandas as pd
from data.process_drugbank import parse_drugbank_xml
from data.process_graph import parse_graph
from data.fetch_data import fetch_data
from data.gain import gain
from data.utils import rmse_loss


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
    print(adj.shape)

    df = df.loc[list(nx_graph.nodes())]
    categorical_feature_df = df[['type', 'groups', 'ATC1', 'ATC2', 'ATC3', 'ATC4', 'ATC5']].copy()
    categorical_feature_df.index = range(len(categorical_feature_df))
    missing_data_df = df[['logP', 'logS', 'psa', 'refractivity', 'polarizability', 'pKa_acidic', 'pKa_basic', 'num_rings']].copy()

    pd.set_option('display.max_columns', None)

    missing_matrix = missing_data_df.to_numpy()
    gain_parameters = {
        'batch_size': 128,
        'hint_rate': 0.9,
        'alpha': 100,
        'iterations': 1000
    }
    imputed_feature_matrix = gain(missing_matrix, gain_parameters)
    imputed_df = pd.DataFrame(imputed_feature_matrix,
                              columns=[
                                  'logP', 'logS', 'psa', 'refractivity', 'polarizability', 'pKa_acidic', 'pKa_basic', 'num_rings'
                              ])
    imputed_df.index = range(len(imputed_df))
    imputed_df = (imputed_df - imputed_df.mean()) / imputed_df.std()

    feature_df = pd.concat([categorical_feature_df, imputed_df], axis=1)
    print(feature_df.describe())

    features = sp.csr_matrix(feature_df)

    return adj, features

# Here we collect the preprocessed data and prepare it to GNN model

import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import networkx as nx 
from torch_geometric.utils.convert import from_networkx
import os
import pickle
from torch_geometric.data import Data


# - X1, A1 are feature set and similarity matrix of Medications
# - X2, A2 are feature set and similarity matrix of Diagnosis
# - X3, A3 are feature set and similarity matrix of Procedures

# - following this approach (https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275)

def reading_pickle(n):
    with open(f'{n}', 'rb') as f:
        data = pd.read_pickle(f)
    numpy_array = np.array(data)
    return numpy_array


def get_edgesList(SM, Patients):
    '''
    SM: similarity matrix,
    return edge_index'''
    N = SM.shape[0]
    edge_list = []
    for i in range(N-1):
        for j in range(i+1, N):
            if SM[i,j]> 0:
                u, v = Patients[i], Patients[j]
                edge_list.append([u, v, SM[i,j]])

    return edge_list



def read_data(my_path):
    XM = reading_pickle(f'{my_path}/X1.pickle')
    XD = reading_pickle(f'{my_path}/X2.pickle')
    XP = reading_pickle(f'{my_path}/X3.pickle')
    # XG = reading_pickle('Data/gender.pkl')
    Y  = reading_pickle(f'{my_path}/Y.pickle')
    Patients  = reading_pickle(f'{my_path}/Patients.pickle')

    AM = reading_pickle(f'{my_path}/A1.pickle')
    AD = reading_pickle(f'{my_path}/A2.pickle')
    AP = reading_pickle(f'{my_path}/A3.pickle')
    return XM, XD, XP, Y, AM, AD, AP, Patients


def get_data(myPath='../../Data/version2'):
    XM, XD, XP, Y, AM, AD, AP, Patients = read_data(myPath)
    # -------------------------------------------
    # Creating Data.X, and Y!

    # Normalization using MIN-MAX
    # XG = np.reshape(XG, (XG.shape[0], 1))
    X = np.concatenate([XM, XP, XD], axis=1)

    # Min-Max Normalization
    # min_vals = np.min(X, axis=0)
    # max_vals = np.max(X, axis=0)

    # X = (X - min_vals) / (max_vals - min_vals)
    # X = np.nan_to_num(X, nan=0)

    # -------------------------------------------
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    # -------------------------------------------

    wm = 0.33
    wd = 0.33
    wp = 0.33

    W =   wm * AM  + wp * AP + wd * AD

    edge_list = get_edgesList(W, Patients)
    G = nx.Graph()
    G.add_nodes_from([
        (Patients[i], {'y': Y[i], 'x': X[i]}) for i in range(len(Patients))
        ])
    G.add_weighted_edges_from(edge_list)

    G = undersampling(G)

    data = from_networkx(G)
    data.num_classes = 2
    data.num_features = X.shape[1]
    print(data)

    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    data = split(data)
    return data


def undersampling(G):
    # Undersampling
    Nodes0 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 0]
    Nodes1 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 1]
    print('Before ', len(Nodes0), len(Nodes1))

    NodesToStay = random.sample(Nodes1, 201)
    G.remove_nodes_from([i for i in Nodes1 if i not in NodesToStay])

    Nodes0 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 0]
    Nodes1 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 1]
    print('After ',len(Nodes0), len(Nodes1))
    return G


def Heter_Undersampling(G):
    # Undersampling patients using Hetereogenous graph
    Nodes0 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 0]
    Nodes1 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 1]
    print('Before ', len(Nodes0), len(Nodes1))

    NodesToStay = random.sample(Nodes0, len(Nodes1))
    G.remove_nodes_from([i for i in Nodes0 if i not in NodesToStay])

    Nodes0 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 0]
    Nodes1 = [node for node, attrs in G.nodes(data=True) if attrs.get('y', None) == 1]
    print('After ',len(Nodes0), len(Nodes1))
    return G

def plot_scatter(X, y):
    # Convert tensor X to a numpy array
    X_np = X.detach().numpy()

    # Create a scatter plot
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y, cmap='viridis')

    # Add labels and title to the plot
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Embedding of the test patients')

    # Show the plot
    plt.show()


# def get_edge_index(SM, th=0, Patients = []):
#     '''
#     SM: similarity matrix,
#     th: threshold for edge weight,
#     return edge_index'''
#     source = []
#     target = []
#     weight = []
#     N = SM.shape[0]
#     for i in range(N-1):
#         for j in range(i+1, N):
#             if SM[i,j]> th:
#                 source.append(Patients[i])
#                 target.append(Patients[j])
#                 weight.append(SM[i,j])

#     return torch.tensor([source, target])


# def get_data(myPath='../../Data/version2'):
#     XM, XD, XP, Y, AM, AD, AP, Patients = read_data(myPath)
#     # -------------------------------------------
#     # Creating Data.X, and Y!

#     # Normalization using MIN-MAX
#     # XG = np.reshape(XG, (XG.shape[0], 1))
#     X = np.concatenate([XM, XP, XD], axis=1)

#     # Min-Max Normalization
#     # min_vals = np.min(X, axis=0)
#     # max_vals = np.max(X, axis=0)

#     # X = (X - min_vals) / (max_vals - min_vals)
#     # X = np.nan_to_num(X, nan=0)

#     # -------------------------------------------
#     X = torch.tensor(X)
#     num_classes = 2
#     num_features = X.shape[1]
#     Y = torch.tensor(Y)
#     # -------------------------------------------

#     wm = 0.10
#     wd = 0.4
#     wp = 0.5

#     W =   wm * AM  + wp * AP + wd * AD

#     e_th =0.0

#     edge_index = get_edge_index(W, e_th, Patients)

#     data = Data(x=X, edge_index = edge_index, y = Y, num_classes = num_classes, num_features = num_features)

#     split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
#     data = split(data)
#     return data






# ---------------------------------------------------------------------------

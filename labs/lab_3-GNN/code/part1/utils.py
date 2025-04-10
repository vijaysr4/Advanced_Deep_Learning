import networkx as nx
import numpy as np
import torch
from random import randint
import scipy

def create_dataset() -> tuple[list[nx.Graph], list[int]]:
    """
    Generate a dataset of random graphs and their corresponding labels.

    The dataset contains two classes of graphs:
    - Class 0: G(n, 0.1) Erdos-Renyi model.
    - Class 1: G(n, 0.4) Erdos-Renyi model.

    Returns:
        tuple[list[nx.Graph], list[int]]: A tuple containing a list of graphs and a list of labels.
    """
    Gs = []
    y = []

    ############## Task 1
    # your code here #
    # Generate Class 0 graphs (G(n, 0.1))
    for _ in range(50):
        num_nodes = randint(10, 20)
        G = nx.fast_gnp_random_graph(num_nodes, 0.1)
        Gs.append(G)
        y.append(0)

    # Generate Class 1 graphs (G(n, 0.4))
    for _ in range(50):
        num_nodes = randint(10, 20)
        G = nx.fast_gnp_random_graph(num_nodes, 0.4)
        Gs.append(G)
        y.append(1)

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx: "scipy.sparse.spmatrix") -> torch.sparse.Tensor:
    """
    Convert a scipy sparse matrix to a PyTorch sparse tensor.

    Args:
        sparse_mx (scipy.sparse.spmatrix): The sparse matrix to convert.

    Returns:
        torch.sparse.Tensor: The converted PyTorch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse_coo_tensor(indices, values, shape)

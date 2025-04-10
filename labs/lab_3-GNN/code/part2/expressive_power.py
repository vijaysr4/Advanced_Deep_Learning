import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor
from scipy.sparse import block_diag

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
# your code here #
##################

# Create 10 cycle graphs with sizes n = 10, 11, ..., 19
cycle_graphs = [nx.cycle_graph(n) for n in range(10, 20)]

# Print the number of nodes for each generated cycle graph
for i, g in enumerate(cycle_graphs):
    print(f"Cycle Graph {i + 1}: {len(g.nodes())} nodes")


############## Task 5
        
##################
# your code here #
##################
# Initialize adjacency matrices and features
adj_list = []
features_list = []
idx_batch_list = []

# Iterate over the cycle graphs created in Task 4
node_offset = 0  # Tracks the offset for nodes across graphs
for graph_idx, G in enumerate(cycle_graphs):
    # Adjacency matrix
    A = nx.adjacency_matrix(G)  # Convert to SciPy sparse matrix
    A = A + sp.eye(A.shape[0])  # Add self-loops
    adj_list.append(A)
    
    # Feature matrix (all ones since no node features are provided)
    num_nodes = A.shape[0]
    features_list.append(np.ones((num_nodes, 1), dtype=np.float32))
    
    # Graph index for each node
    idx_batch_list.extend([graph_idx] * num_nodes)

# Create block-diagonal adjacency matrix
adj_block = sp.block_diag(adj_list).tocsr()

# Combine features into a single matrix
features_batch = np.vstack(features_list)

# Convert to PyTorch tensors
adj_batch_torch = sparse_mx_to_torch_sparse_tensor(adj_block).to(device)
features_batch_torch = torch.FloatTensor(features_batch).to(device)
idx_batch_torch = torch.LongTensor(idx_batch_list).to(device)

# Print to verify correctness
print(f"Adjacency Matrix Shape: {adj_batch_torch.shape}")
print(f"Feature Matrix Shape: {features_batch_torch.shape}")
print(f"Graph Index Vector Shape: {idx_batch_torch.shape}")
'''
Adjacency Matrix Shape: torch.Size([145, 145])
Feature Matrix Shape: torch.Size([145, 1])
Graph Index Vector Shape: torch.Size([145])
'''

############## Task 8
        
##################
# your code here #

# Initialize a GNN with mean operator for both neighborhood aggregation and readout
gnn_mean = GNN(
    input_dim=1,          # Input feature dimension
    hidden_dim=hidden_dim,  # Hidden layer dimension
    output_dim=output_dim,  # Output feature dimension
    neighbor_aggr='mean',   # Neighborhood aggregation method
    readout='mean',         # Readout function
    dropout=dropout         # Dropout rate
).to(device)

# Perform a feedforward pass and compute graph representations
with torch.no_grad():  # Disable gradient computation
    graph_representations_mean = gnn_mean(features_batch_torch, adj_batch_torch, idx_batch_torch)

print("Graph Representations using mean aggregation for both neighborhood and readout:")
print(graph_representations_mean)
'''
Graph Representations using mean aggregation for both neighborhood and readout:
tensor([[-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659],
        [-0.3697, -0.3108,  0.4701, -0.3659]], device='cuda:0')
'''

# Initialize another GNN with sum aggregation for neighborhood and mean for readout
gnn_sum_mean = GNN(
    input_dim=1,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    neighbor_aggr='sum',    # Sum for neighborhood aggregation
    readout='mean',         # Mean for readout
    dropout=dropout
).to(device)

# Perform a feedforward pass and compute graph representations
with torch.no_grad():
    graph_representations_sum_mean = gnn_sum_mean(features_batch_torch, adj_batch_torch, idx_batch_torch)

print("\nGraph Representations using sum aggregation for neighborhood and mean for readout:")
print(graph_representations_sum_mean)
'''
Graph Representations using sum aggregation for neighborhood and mean for readout:
tensor([[-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869],
        [-2.5046, -0.4506, -1.0141,  0.3869]], device='cuda:0')
'''
# Initialize another GNN with mean aggregation for neighborhood and sum for readout
gnn_mean_sum = GNN(
    input_dim=1,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    neighbor_aggr='mean',   # Mean for neighborhood aggregation
    readout='sum',          # Sum for readout
    dropout=dropout
).to(device)

# Perform a feedforward pass and compute graph representations
with torch.no_grad():
    graph_representations_mean_sum = gnn_mean_sum(features_batch_torch, adj_batch_torch, idx_batch_torch)

print("\nGraph Representations using mean aggregation for neighborhood and sum for readout:")
print(graph_representations_mean_sum)
'''
Output:
Graph Representations using mean aggregation for neighborhood and sum for readout:
tensor([[-0.8938,  1.0392, -0.5993, -0.7140],
        [-0.9824,  1.1386, -0.6725, -0.7688],
        [-1.0710,  1.2381, -0.7457, -0.8236],
        [-1.1595,  1.3375, -0.8189, -0.8784],
        [-1.2481,  1.4370, -0.8921, -0.9332],
        [-1.3367,  1.5364, -0.9652, -0.9880],
        [-1.4253,  1.6359, -1.0384, -1.0428],
        [-1.5139,  1.7353, -1.1116, -1.0976],
        [-1.6024,  1.8348, -1.1848, -1.1524],
        [-1.6910,  1.9342, -1.2580, -1.2072]], device='cuda:0')
'''

# Initialize another GNN with sum aggregation for both neighborhood and readout
gnn_sum_sum = GNN(
    input_dim=1,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    neighbor_aggr='sum',    # Sum for neighborhood aggregation
    readout='sum',          # Sum for readout
    dropout=dropout
).to(device)

# Perform a feedforward pass and compute graph representations
with torch.no_grad():
    graph_representations_sum_sum = gnn_sum_sum(features_batch_torch, adj_batch_torch, idx_batch_torch)

print("\nGraph Representations using sum aggregation for both neighborhood and readout:")
print(graph_representations_sum_sum)
'''
Output:
Graph Representations using sum aggregation for both neighborhood and readout:
tensor([[-3.0624,  3.9958, -1.8124,  6.4134],
        [-3.3562,  4.3965, -1.9997,  7.0451],
        [-3.6500,  4.7971, -2.1871,  7.6768],
        [-3.9439,  5.1978, -2.3744,  8.3085],
        [-4.2377,  5.5985, -2.5617,  8.9402],
        [-4.5316,  5.9992, -2.7490,  9.5718],
        [-4.8254,  6.3999, -2.9363, 10.2035],
        [-5.1192,  6.8005, -3.1237, 10.8352],
        [-5.4131,  7.2012, -3.3110, 11.4669],
        [-5.7069,  7.6019, -3.4983, 12.0986]], device='cuda:0')
'''


############## Task 9
        
##################
# your code here #

# Create graph G1 (two connected components)
G1 = nx.Graph()
# First component: A triangle (3-cycle)
G1.add_edges_from([(0, 1), (1, 2), (2, 0)])
# Second component: Another triangle (disconnected)
G1.add_edges_from([(3, 4), (4, 5), (5, 3)])

# Create graph G2 (a 6-cycle)
G2 = nx.cycle_graph(6)

# Print the details of the graphs
print("Graph G1:")
print(f"Number of nodes: {G1.number_of_nodes()}, Number of edges: {G1.number_of_edges()}")
print(f"Edges: {list(G1.edges())}\n")

print("Graph G2:")
print(f"Number of nodes: {G2.number_of_nodes()}, Number of edges: {G2.number_of_edges()}")
print(f"Edges: {list(G2.edges())}")

# Adjacency matrices and node features
adj_matrices = [nx.adjacency_matrix(G) for G in [G1, G2]]
sparse_block_adj = block_diag(adj_matrices, format="csr")

# Node features: All nodes initialized with 1s
num_nodes = sparse_block_adj.shape[0]
node_features = np.ones((num_nodes, 1), dtype=np.float32)

# Graph membership vector
graph_membership = []
for graph_idx, G in enumerate([G1, G2]):
    num_nodes_in_graph = G.number_of_nodes()
    graph_membership.extend([graph_idx] * num_nodes_in_graph)
graph_membership = np.array(graph_membership)

# Convert data to PyTorch tensors
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Converts a SciPy sparse matrix to a PyTorch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()  # Convert to COO format
    indices = torch.LongTensor(np.vstack((sparse_mx.row, sparse_mx.col)))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

torch_sparse_block_adj = sparse_mx_to_torch_sparse_tensor(sparse_block_adj)
torch_node_features = torch.FloatTensor(node_features)
torch_graph_membership = torch.LongTensor(graph_membership)

# Print details about processed data
print("\nProcessed Data:")
print(f"Sparse Block Diagonal Adjacency Matrix Shape: {sparse_block_adj.shape}")
print(f"Node Features Shape: {torch_node_features.shape}")
print(f"Graph Membership Vector Shape: {torch_graph_membership.shape}")

# Visualize the graphs (Optional, requires matplotlib)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    nx.draw(G1, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
    plt.title("Graph G1")

    plt.subplot(122)
    nx.draw(G2, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=700)
    plt.title("Graph G2")

    plt.show()
except ImportError:
    print("Matplotlib is not installed. Skipping visualization.")

############## Task 10
        
##################
# your code here #

# Create block-diagonal adjacency matrix for G1 and G2
adj_matrices = [nx.adjacency_matrix(G) for G in [G1, G2]]  # Adjacency matrices for G1 and G2
sparse_block_adj = block_diag(adj_matrices, format="csr")  # Create block-diagonal adjacency matrix

# Initialize feature matrix for all nodes (all values set to 1)
num_nodes = sparse_block_adj.shape[0]
node_features = np.ones((num_nodes, 1), dtype=np.float32)  # All nodes initialized with the same value (1)

# Create a graph membership vector
graph_membership = []
for graph_idx, G in enumerate([G1, G2]):  # Loop over G1 and G2
    num_nodes_in_graph = G.number_of_nodes()
    graph_membership.extend([graph_idx] * num_nodes_in_graph)  # Assign graph index to nodes
graph_membership = np.array(graph_membership)

# Convert data to PyTorch tensors
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Converts a SciPy sparse matrix to a PyTorch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()  # Convert to COO format
    indices = torch.LongTensor(np.vstack((sparse_mx.row, sparse_mx.col)))  # Row and column indices
    values = torch.FloatTensor(sparse_mx.data)  # Non-zero values
    shape = torch.Size(sparse_mx.shape)  # Shape of the matrix
    return torch.sparse.FloatTensor(indices, values, shape)

# Convert adjacency matrix to PyTorch sparse tensor
torch_sparse_block_adj = sparse_mx_to_torch_sparse_tensor(sparse_block_adj)

# Convert feature matrix and graph membership vector to PyTorch tensors
torch_node_features = torch.FloatTensor(node_features)
torch_graph_membership = torch.LongTensor(graph_membership)

# Print the shapes of the processed data
print("\nProcessed Data:")
print(f"Sparse Block Diagonal Adjacency Matrix Shape: {torch_sparse_block_adj.shape}")
print(f"Node Features Shape: {torch_node_features.shape}")
print(f"Graph Membership Vector Shape: {torch_graph_membership.shape}")


############## Task 11
        
##################
# your code here #

# Initialize a GNN with sum aggregation for both neighborhood aggregation and readout
gnn_sum = GNN(
    input_dim=1,          # Input feature dimension
    hidden_dim=hidden_dim,  # Hidden layer dimension
    output_dim=output_dim,  # Output feature dimension
    neighbor_aggr='sum',    # Sum for neighborhood aggregation
    readout='sum',          # Sum for readout
    dropout=dropout         # Dropout rate
).to(device)

# Ensure all tensors are on the same device
torch_sparse_block_adj = torch_sparse_block_adj.to(device)
torch_node_features = torch_node_features.to(device)
torch_graph_membership = torch_graph_membership.to(device)

# Perform a feedforward pass and compute graph representations for G1 and G2
gnn_sum.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    graph_representations = gnn_sum(torch_node_features, torch_sparse_block_adj, torch_graph_membership)

# Print the vector representations for G1 and G2
print("\nTask 11 Output:")
print("Graph Representations using sum aggregation for both neighborhood and readout:")
for i, rep in enumerate(graph_representations):
    print(f"Graph G{i + 1} representation: {rep.cpu().numpy()}")
'''
Task 11 Output:
Graph Representations using sum aggregation for both neighborhood and readout:
Graph G1 representation: [ 4.193077  -1.7683479 -6.670186  -3.3090365]
Graph G2 representation: [ 4.193077  -1.7683479 -6.670186  -3.3090365]
'''

'''
Conclusion:
    
The vectors for G1 and G2 are identical, 
indicating that the GNN is unable to differentiate between these two non-isomorphic graphs.
It is because of the readout methods, 
resulting in the loss of structural information. 
'''


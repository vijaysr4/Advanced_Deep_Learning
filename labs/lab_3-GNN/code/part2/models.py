import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassing(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, neighbor_aggr: str):
        """
        Message passing layer for aggregating information from neighbors.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            neighbor_aggr (str): Aggregation method ('sum' or 'mean').
        """
        super(MessagePassing, self).__init__()
        self.neighbor_aggr = neighbor_aggr
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        """
        Forward pass for message passing.

        Args:
            x (torch.Tensor): Node feature matrix of shape (N, input_dim).
            adj (torch.sparse.Tensor): Sparse adjacency matrix of shape (N, N).

        Returns:
            torch.Tensor: Updated node features of shape (N, output_dim).
        """
        ############## Task 6
    
        ##################
        # your code here #

        # Transform node features
        x_node = self.fc1(x)

        # Aggregate messages from neighbors
        m = torch.spmm(adj, x)  # Sparse matrix multiplication (N x input_dim)

        # Transform aggregated messages
        m = self.fc2(m)
        
        if self.neighbor_aggr == 'sum':
            output = x_node + m
        elif self.neighbor_aggr == 'mean':
            deg = torch.spmm(adj, torch.ones(x.size(0),1, device=x.device))
            output = x_node + torch.div(m, deg)
            
        return output



class GNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, neighbor_aggr: str, readout: str, dropout: float):
        """
        Graph Neural Network consisting of two message passing layers and a readout function.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden features.
            output_dim (int): Dimension of output features.
            neighbor_aggr (str): Aggregation method ('sum' or 'mean').
            readout (str): Readout function ('sum' or 'mean').
            dropout (float): Dropout rate for regularization.
        """
        super(GNN, self).__init__()
        self.readout = readout
        self.mp1 = MessagePassing(input_dim, hidden_dim, neighbor_aggr)
        self.mp2 = MessagePassing(hidden_dim, hidden_dim, neighbor_aggr)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GNN.

        Args:
            x (torch.Tensor): Node feature matrix of shape (N, input_dim).
            adj (torch.sparse.Tensor): Sparse adjacency matrix of shape (N, N).
            idx (torch.Tensor): Graph membership vector indicating which graph each node belongs to.

        Returns:
            torch.Tensor: Output feature vectors for the graphs, of shape (num_graphs, output_dim).
        """
        
        ############## Task 7
    
        ##################
        # your code here #
        
        # First message passing layer
        x = self.mp1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)

        # Second message passing layer
        x = self.mp2(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        
        if self.readout == 'sum':
            idx = idx.unsqueeze(1).repeat(1, x.size(1))
            out = torch.zeros(torch.max(idx)+1, x.size(1), device=x.device)
            out = out.scatter_add_(0, idx, x) 
        elif self.readout == 'mean':
            idx = idx.unsqueeze(1).repeat(1, x.size(1))
            out = torch.zeros(torch.max(idx)+1, x.size(1), device=x.device)
            out = out.scatter_add_(0, idx, x)
            count = torch.zeros(torch.max(idx)+1, x.size(1), device=x.device)
            count = count.scatter_add_(0, idx, torch.ones_like(x, device=x.device))
            out = torch.div(out, count)
            
        ############## Task 7
    
        ##################
        # your code here #
        
        # Fully connected layer
        out = self.fc(out)
        
        return out

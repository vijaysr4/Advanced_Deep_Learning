import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, hidden_dim_3: int, n_class: int):
        """
        Initialize the GNN model.

        Args:
            input_dim (int): The dimension of input node features.
            hidden_dim_1 (int): The number of hidden units in the first message-passing layer.
            hidden_dim_2 (int): The number of hidden units in the second message-passing layer.
            hidden_dim_3 (int): The number of hidden units in the first fully connected layer.
            n_class (int): The number of output classes.
        """
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in: Tensor, adj: Tensor, idx: Tensor) -> Tensor:
        """
        Perform a forward pass of the GNN.

        Args:
            x_in (Tensor): Node feature matrix of shape (N, d), where N is the number of nodes 
                           and d is the input feature dimension.
            adj (Tensor): Sparse adjacency matrix of shape (N, N).
            idx (Tensor): Batch assignment vector mapping nodes to graphs, of shape (N,).

        Returns:
            Tensor: Log-softmax output of shape (num_graphs, n_class), where num_graphs is the 
                    number of graphs in the batch and n_class is the number of classes.
        """
        
        ############## Task 2
        # your code here #
        # Message Passing Layer 1
        x = torch.spmm(adj, x_in)  # Apply adjacency matrix to input features
        x = self.fc1(x)  # Linear transformation W^0
        x = self.relu(x)  # ReLU activation

        # Message Passing Layer 2
        x = torch.spmm(adj, x)  # Apply adjacency matrix to transformed features
        x = self.fc2(x)  # Linear transformation W^1
        x = self.relu(x)  # ReLU activation
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1), device=x_in.device)
        out = out.scatter_add_(0, idx, x) 
        
        # your code here #
        # Fully connected layers
        out = self.fc3(out)  # Linear transformation W^2
        out = self.relu(out)  # ReLU activation
        out = self.fc4(out)  # Linear transformation W^3

        return F.log_softmax(out, dim=1)

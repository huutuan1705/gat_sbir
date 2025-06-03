import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional layer, similar to Kipf & Welling (2017).
    Assumes features are (Num_Nodes, In_Features) and adj is (Num_Nodes, Num_Nodes).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            bias (bool): Whether to use a bias term.
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weight and bias."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features (torch.Tensor): Input node features, shape (N, in_features),
                                           where N is the number of nodes (labels).
            adj (torch.Tensor): Adjacency matrix (potentially normalized),
                                shape (N, N). Can be sparse or dense.
                                If sparse, torch.spmm will be used. Otherwise, torch.mm.

        Returns:
            torch.Tensor: Output node features, shape (N, out_features).
        """
        support = torch.mm(input_features, self.weight) # XW
        if adj.is_sparse:
            output = torch.spmm(adj, support) # A(XW) for sparse adj
        else:
            output = torch.mm(adj, support) # A(XW) for dense adj
            
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) module.
    The paper mentions a two-layer GCN operating on latent graph embeddings (label features). [cite: 155]
    """
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float = 0.1, num_layers: int = 2):
        """
        Args:
            nfeat (int): Number of input features per node (e.g., GloVe embedding dim).
            nhid (int): Number of hidden units in the GCN layer(s).
            nclass (int): Number of output features per node (dimension of GCN processed label features).
            dropout (float): Dropout rate.
            num_layers (int): Number of GCN layers. Paper suggests 2. [cite: 155]
        """
        super(GCN, self).__init__()
        
        if num_layers < 1:
            raise ValueError("Number of GCN layers must be at least 1.")

        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer GCN: input directly to output
            self.layers.append(GraphConvolution(nfeat, nclass))
        else:
            # First layer: input to hidden
            self.layers.append(GraphConvolution(nfeat, nhid))
            # Intermediate hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(GraphConvolution(nhid, nhid))
            # Output layer: hidden to output
            self.layers.append(GraphConvolution(nhid, nclass))
            
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GCN model.

        Args:
            x (torch.Tensor): Input features for nodes (labels), shape (N, nfeat).
            adj (torch.Tensor): Adjacency matrix for the graph, shape (N, N).
                                This should ideally be a normalized adjacency matrix
                                (e.g., D^-0.5 * A_hat * D^-0.5).

        Returns:
            torch.Tensor: Output features for nodes (processed label embeddings),
                          shape (N, nclass).
        """
        for i, layer in enumerate(self.layers[:-1]): # All layers except the last one
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
        
        # Last layer
        x = self.layers[-1](x, adj)
        
        return x
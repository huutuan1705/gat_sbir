import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 dropout: float, alpha: float = 0.2, concat_heads: bool = True):
        """
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node (for each head).
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention coefficients and output features.
            alpha (float): Alpha for the LeakyReLU activation.
            concat_heads (bool): If True, the outputs of n_heads are concatenated.
                                 Otherwise, they are averaged.
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout
        self.alpha = alpha

        # Linear transformation for each head (applied to input node features)
        # W_k: (in_features, out_features) for each head k
        self.W = nn.Parameter(torch.empty(size=(in_features, n_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters 'a' for each head
        # a_k: (2 * out_features, 1) for each head k
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, n_heads)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.concat_heads:
            self.output_dim = out_features * n_heads
        else:
            self.output_dim = out_features
            
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Graph Attention Layer.

        Args:
            node_features (torch.Tensor): Input node features, shape (N, in_features),
                                          where N is the number of nodes.
            adj_matrix (torch.Tensor): Adjacency matrix (can be dense or sparse, binary or weighted),
                                     shape (N, N). Used to mask attention to only consider neighbors.
                                     A value > 0 indicates an edge.

        Returns:
            torch.Tensor: Output node features, shape (N, n_heads * out_features) if concat_heads is True,
                          else (N, out_features).
        """
        N = node_features.size(0) # Number of nodes
        h_transformed = torch.mm(node_features, self.W)
        h_transformed = h_transformed.view(N, self.n_heads, self.out_features)
        
        # Wh_i for all i, repeated N times: (N*N, n_heads, out_features)
        h_left = h_transformed.repeat_interleave(N, dim=0)
        # Wh_j for all j, tiled N times: (N*N, n_heads, out_features)
        h_right = h_transformed.repeat(N, 1, 1)
        
        # Concatenate [Wh_i || Wh_j]: (N*N, n_heads, 2 * out_features)
        h_concat = torch.cat([h_left, h_right], dim=-1)
        
        # Apply attention parameters 'a'
        # self.a: (n_heads, 2 * out_features) -> (n_heads, 2 * out_features, 1) for matmul
        # e_unnormalized: (N*N, n_heads, 1)
        
        # print("h_concat shape: ", h_concat.shape) # [361, 4, 512]
        # print("self.a shape: ", self.a.shape) # [4, 512]
        
        e_unnormalized = torch.matmul(h_concat, self.a.unsqueeze(-1))
        e_unnormalized = self.leakyrelu(e_unnormalized.squeeze(-1)) # (N*N, n_heads)
        e_unnormalized = e_unnormalized.view(N, N, self.n_heads) # (N, N, n_heads)
        
        # Masking: only attend to neighbors (as defined by adj_matrix)
        # Create a mask where non-neighbors have -infinity attention scores
        zero_vec = -9e15 * torch.ones_like(e_unnormalized)
        # adj_matrix: (N, N). Needs to be broadcastable to (N, N, n_heads)
        adj_mask = adj_matrix.unsqueeze(-1) # (N, N, 1)
        
        attention_scores = torch.where(adj_mask > 0, e_unnormalized, zero_vec)
        attention_probs = F.softmax(attention_scores, dim=1) # Softmax over columns (j for fixed i)
        attention_probs = F.dropout(attention_probs, self.dropout_rate, training=self.training) # (N, N, n_heads)

        # Aggregate features from neighbors using attention probabilities
        # attention_probs: (N, N, n_heads), h_transformed: (N, n_heads, out_features)
        # For each head, compute weighted sum: output_h_k = sum_j (alpha_ij_k * Wh_j_k)
        # Transpose for bmm: attn (n_heads, N, N), h_trans (n_heads, N, out_features)
        output_h = torch.bmm(attention_probs.permute(2, 0, 1), h_transformed.permute(1, 0, 2))
        output_h = output_h.permute(1, 0, 2) # (N, n_heads, out_features)

        if self.concat_heads:
            # Concatenate features from all heads
            output_h = output_h.contiguous().view(N, -1) # (N, n_heads * out_features)
            return F.elu(output_h) # As in the original GAT paper
        else:
            # Average features from all heads
            output_h = torch.mean(output_h, dim=1) # (N, out_features)
            return F.elu(output_h)
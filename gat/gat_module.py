import torch
import torch.nn as nn
import torch.nn.functional as F

from gat.gat_layer import GraphAttentionLayer

class FGSBIR_GAT(nn.Module):
    def __init__(self, visual_feature_dim: int, 
                 gat_out_per_head_dim: int, n_gat_heads: int, 
                 gcn_label_feature_dim: int,
                 final_embedding_dim: int, num_gat_layers: int = 1,
                 dropout: float = 0.1, alpha_gat: float = 0.2):
        super(FGSBIR_GAT, self).__init__()
        self.visual_feature_dim = visual_feature_dim
        self.gcn_label_feature_dim = gcn_label_feature_dim
        
        self.gat_input_dim = gcn_label_feature_dim + self.visual_feature_dim
        
        self.gat_layers = nn.ModuleList()
        current_dim = self.gat_input_dim
        
        for i in range(num_gat_layers):
            gat_layer = GraphAttentionLayer(
                in_features=current_dim,
                out_features=gat_out_per_head_dim,
                n_heads=n_gat_heads,
                dropout=dropout,
                alpha=alpha_gat,
                concat_heads=True # Always concat for intermediate layers
            )
            self.gat_layers.append(gat_layer)
            current_dim = gat_layer.output_dim # Update current_dim for the next layer
        
        self.attention_pooling_query = nn.Linear(current_dim, 1, bias=False) # Learnable query for pooling
        
        # Final projection to the desired common search space embedding dimension
        self.final_fc = nn.Linear(current_dim, final_embedding_dim)
        
    def forward(self, visual_features: torch.Tensor, labels_features: torch.Tensor, label_adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features (torch.Tensor): (B, D_vis_global)
            label_features (torch.Tensor): (N_labels, D_gcn_label_feat) - output from GCN module
            label_adj_matrix (torch.Tensor): (N_labels, N_labels) - adjacency for label graph

        Returns:
            torch.Tensor: Final image embedding for the common search space, shape (B, final_embedding_dim)
        """
        B = visual_features.size(0)
        N_labels = labels_features.size(0)
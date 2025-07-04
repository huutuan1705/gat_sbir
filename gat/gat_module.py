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
        
        # 2. Contextualize label features with image features
        # Expand visual features for each label: (B, D_vis_projected) -> (B, N_labels, D_vis_projected)
        image_context_expanded = visual_features.unsqueeze(1).repeat(1, N_labels, 1)
        
        # Expand GCN label features for each image in batch: (N_labels, D_gcn) -> (B, N_labels, D_gcn)
        all_gcn_label_features_expanded = labels_features.unsqueeze(0).repeat(B, 1, 1)
        
        # Concatenate image context with each label feature: (B, N_labels, D_gcn + D_vis_projected)
        contextualized_label_node_features = torch.cat(
            (all_gcn_label_features_expanded, image_context_expanded), dim=-1
        )
        
        # 3. Apply GAT layers (processing each image in the batch independently)
        batch_pooled_features = []
        for i in range(B): # Iterate over each image in the batch
            # current_image_label_features: (N_labels, self.gat_input_dim)
            current_image_label_features = contextualized_label_node_features[i] 
            
            # Pass through GAT layers
            gat_processed_features = current_image_label_features
            for gat_layer in self.gat_layers:
                gat_processed_features = gat_layer(gat_processed_features, label_adj_matrix)
            # gat_processed_features: (N_labels, gat_output_dim_concatenated)
            
            # 4. Attention-based Pooling over label features for this image
            # This pools the N_labels features into a single feature vector for the image.
            # att_weights_pool: (N_labels, 1) -> (N_labels)
            att_weights_pool = self.attention_pooling_query(gat_processed_features).squeeze(-1)
            att_weights_pool = F.softmax(att_weights_pool, dim=0) # (N_labels)
            
            # pooled_features_for_image: (gat_output_dim_concatenated)
            pooled_features_for_image = torch.sum(gat_processed_features * att_weights_pool.unsqueeze(-1), dim=0)
            batch_pooled_features.append(pooled_features_for_image)
            
        # Stack pooled features for all images in the batch
        final_batch_features = torch.stack(batch_pooled_features, dim=0) # (B, gat_output_dim_concatenated)
        
        # 5. Final projection to common search space embedding
        search_space_embedding = self.final_fc(final_batch_features) # (B, final_embedding_dim)
        
        return search_space_embedding
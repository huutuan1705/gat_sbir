import torch
import torch.nn as nn
from baseline.backbones import InceptionV3
from baseline.attention import SelfAttention, Linear_global

from gat.label_embedding import LabelEmbeddings
from gat.gat_module import FGSBIR_GAT
from gat.gcn_module import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-label Information-Guided Graph (MIGG) model for on-the-fly FG-SBIR
class MIGG(nn.Module):
    def __init__(self, num_classes: int, config: dict, args):
        super(MIGG, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.args = args
        
        self.sample_embedding_network = InceptionV3(args=args).to(device)
        self.attention = SelfAttention(args).to(device)
        self.linear = Linear_global(feature_num=self.args.output_size).to(device)
        
        self.sketch_embedding_network = InceptionV3(args=args).to(device)
        self.sketch_attention = SelfAttention(args).to(device)
        self.sketch_linear = Linear_global(feature_num=self.args.output_size).to(device)
        
        self.label_embedder = LabelEmbeddings(
            num_classes=num_classes,
            embedding_dim=self.config['label_embeddings']['embedding_dim'],
            glove_file_path=self.config['label_embeddings'].get('glove_file_path'), # Optional
            label_names=self.config['label_embeddings'].get('label_names')         # Optional
        )
        
        self.gcn_module = GCN(
            nfeat=self.config['label_embeddings']['embedding_dim'], # Input from label embedder
            nhid=self.config['gcn']['hidden_dim'],
            nclass=self.config['gcn']['output_dim'], # Output dim of GCN-processed label features
            dropout=self.config['gcn']['dropout'],
            num_layers=self.config['gcn']['num_layers']
        )
        
        self.image_gat_fusion = FGSBIR_GAT(
            visual_feature_dim=self.config['visual_combiner']['projection_dim'],
            gcn_label_feature_dim=self.config['gcn']['output_dim'],
            gat_out_per_head_dim=self.config['gat']['out_per_head_dim'],
            n_gat_heads=self.config['gat']['num_heads'],
            final_embedding_dim=self.config['gat']['final_embedding_dim'], # This is the search space dim
            num_gat_layers=self.config['gat']['num_layers'],
            dropout=self.config['gat']['dropout'],
            alpha_gat=self.config['gat']['alpha_gat']
        )
        
        search_space_embedding_dim = self.config['gat']['final_embedding_dim']
        self.classifier_head = nn.Linear(search_space_embedding_dim, num_classes)
        
    def forward(self, images: torch.Tensor,
                all_label_indices: torch.Tensor, # e.g., torch.arange(self.num_classes)
                label_adj_matrix: torch.Tensor   # Adjacency matrix for the label graph
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Batch of input images, shape (B, C, H, W).
            all_label_indices (torch.Tensor): Tensor of indices for all labels, e.g., torch.arange(num_classes).
                                             Used to fetch all label embeddings for GCN.
            label_adj_matrix (torch.Tensor): Adjacency matrix for the label graph, shape (N_labels, N_labels).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - search_space_embeddings (torch.Tensor): Final embeddings for the common search space,
                                                          shape (B, search_space_embedding_dim).
                - prediction_scores (torch.Tensor): Logits for each class/label for each image,
                                                    shape (B, num_classes).
                - gcn_processed_label_features (torch.Tensor): Output from the GCN module for all labels,
                                                               shape (N_labels, gcn_output_dim).
                                                               Useful for L_GCN loss component.
        """
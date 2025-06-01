import torch
import torch.nn as nn
import torch.nn.functional as F

from gat.gat_layer import GraphAttentionLayer

class FGSBIR_GAT(nn.Module):
    def __init__(self, visual_feature_dim: int, 
                 gat_out_per_head_dim: int, n_gat_heads: int,
                 final_embedding_dim: int, num_gat_layers: int = 1,
                 dropout: float = 0.1, alpha_gat: float = 0.2):
        super(FGSBIR_GAT, self).__init__()
        
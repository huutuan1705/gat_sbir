import torch
import torch.nn as nn
from baseline.backbones import InceptionV3
from baseline.attention import SelfAttention, Linear_global

from gat.label_embedding import LabelEmbeddings

# Multi-modal Information-Guided Graph (MIGG) for on-the-fly FG-SBIR
class MIGG(nn.Module):
    def __init__(self, num_classes: int, config: dict, args):
        super(MIGG, self).__init__()
        self.num_classes = num_classes
        self.config = config
        
        self.label_embedder = LabelEmbeddings(
            num_classes=num_classes,
            embedding_dim=self.config['label_embeddings']['embedding_dim'],
            glove_file_path=self.config['label_embeddings'].get('glove_file_path'), # Optional
            label_names=self.config['label_embeddings'].get('label_names')         # Optional
        )
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader


def get_label_adjacency_matrix(train_loader: DataLoader, num_classes: int, device: torch.device, threshold_type="dynamic", cooccurrence_threshold=0.01):
    """
    Computes a normalized adjacency matrix for labels based on co-occurrence in the training data.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
                                   Assumes it yields (images, multi_hot_labels, ...)
                                   or (images, single_label_indices, ...) for CIFAR-10.
        num_classes (int): Total number of classes.
        device (torch.device): Device to put the matrix on.
        threshold_type (str): "fixed" or "dynamic". If "dynamic", threshold is based on mean co-occurrence.
        cooccurrence_threshold (float): Threshold for creating an edge if type is "fixed",
                                        or a multiplier if "dynamic".

    Returns:
        torch.Tensor: Normalized adjacency matrix (sparse or dense based on GCN implementation needs).
    """
    print("Computing label co-occurrence adjacency matrix...")
    cooccurrence_counts = np.zeros((num_classes, num_classes), dtype=np.float32)
    num_samples_processed = 0

    for batch_idx, data_batch in enumerate(train_loader):
        *_, multi_hot_labels = data_batch 
        print(multi_hot_labels)
        multi_hot_labels = multi_hot_labels.cpu().numpy()

        # Efficiently compute co-occurrences for the batch
        # For each sample, find pairs of active labels
        for i in range(multi_hot_labels.shape[0]): # Iterate over samples in batch
            active_indices = np.where(multi_hot_labels[i] == 1)[0]
            for j1_idx in range(len(active_indices)):
                for j2_idx in range(j1_idx, len(active_indices)): # Include self-loops in counts initially
                    u, v = active_indices[j1_idx], active_indices[j2_idx]
                    cooccurrence_counts[u, v] += 1
                    if u != v:
                        cooccurrence_counts[v, u] += 1 # Ensure symmetry
        num_samples_processed += multi_hot_labels.shape[0]
        if batch_idx % 100 == 0 :
             print(f"  Processed {num_samples_processed} samples for co-occurrence...")
    
    print(f"Finished processing {num_samples_processed} samples for co-occurrence.")

    # Create adjacency matrix A based on thresholded co-occurrence
    if threshold_type == "dynamic":
        # Threshold based on mean of non-diagonal co-occurrences
        upper_tri_cooccur = cooccurrence_counts[np.triu_indices(num_classes, k=1)]
        if len(upper_tri_cooccur) > 0 and np.mean(upper_tri_cooccur) > 0:
            dynamic_threshold = np.mean(upper_tri_cooccur) * cooccurrence_threshold
        else: # Fallback if no co-occurrences or all zeros
            dynamic_threshold = 0.1 # A small default to ensure some connectivity if needed
        print(f"  Using dynamic co-occurrence threshold: {dynamic_threshold:.4f}")
        adj_matrix_A = (cooccurrence_counts >= dynamic_threshold).astype(np.float32)
    else: # fixed threshold
        print(f"  Using fixed co-occurrence threshold: {cooccurrence_threshold}")
        # Normalize by number of samples to get probabilities if threshold is probability based
        # cooccurrence_probs = cooccurrence_counts / num_samples_processed 
        # adj_matrix_A = (cooccurrence_probs >= cooccurrence_threshold).astype(np.float32)
        # Or, if threshold is count-based:
        adj_matrix_A = (cooccurrence_counts >= cooccurrence_threshold).astype(np.float32)

    # Add self-loops: A_hat = A + I
    adj_hat = adj_matrix_A + np.eye(num_classes, dtype=np.float32)
    adj_hat = np.clip(adj_hat, 0, 1) # Ensure it's binary if A had self-loops from co-occurrence

    # Normalize: D_hat^(-0.5) * A_hat * D_hat^(-0.5)
    D_hat_diag = np.sum(adj_hat, axis=1)
    D_hat_inv_sqrt_diag = np.power(D_hat_diag, -0.5)
    D_hat_inv_sqrt_diag[np.isinf(D_hat_inv_sqrt_diag) | np.isnan(D_hat_inv_sqrt_diag)] = 0.0
    D_hat_inv_sqrt = np.diag(D_hat_inv_sqrt_diag)

    normalized_adj = D_hat_inv_sqrt @ adj_hat @ D_hat_inv_sqrt
    
    print("Label adjacency matrix computed and normalized.")
    # The GCN module in the prototype used torch.spmm for sparse adj.
    # Convert to sparse tensor for GCN.
    normalized_adj_tensor = torch.from_numpy(normalized_adj).to_sparse().to(device)
    return normalized_adj_tensor

def read_labels(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        label_names = header[1:]
    return label_names

def get_model_config(glove_dim=300, csv_path=''):
    label_names_list = read_labels(csv_path)
    return {
        'label_embeddings': {
            'embedding_dim': glove_dim,
            'glove_file_path': '/kaggle/input/glove/pytorch/default/1/glove.6B.300d.txt', # Placeholder
            'label_names': label_names_list # e.g. ['cat', 'dog', 'using_phone', ...]
        },
        'gcn': {
            'hidden_dim': 512,
            'output_dim': 1024, # GCN output label feature dim
            'dropout': 0.1,
            'num_layers': 2
        },
        'global_branch': {
            'backbone_name': 'resnet101',
            'pretrained': True,
            'cnn_output_features_dim': 2048 # Native ResNet101 output
        },
        'local_branch': { # Simplified: uses same backbone as global
            'backbone_name': 'resnet101',
            'pretrained': True,
            'cnn_output_features_dim': 2048
        },
        'visual_combiner': { # This key was used in GLGM __init__ but not used if GAT takes global/local separately
            'projection_dim': 1024, # Example, if global+local were combined then projected before GAT
            'dropout': 0.1
        },
        'gat': { # For ImageContextualizedLabelGAT
            # visual_feature_dim is taken from global_branch.cnn_output_features_dim by GAT if only global is used for projection
            # gcn_label_feature_dim is taken from gcn.output_dim by GAT
            'hidden_dim_Noeudefined_internally': 0, # Placeholder, GAT calculates its true input dim
            'out_per_head_dim': 256,
            'num_heads': 4,
            'final_embedding_dim': 512, # Dimension of common search space
            'num_layers': 2, # Number of GAT layers stacked in ImageContextualizedLabelGAT
            'dropout': 0.1,
            'alpha_gat': 0.2
        },
        'classifier': {
            # Classifier input dim is gat.final_embedding_dim
            # Classifier output dim is num_classes
        }
    }
    
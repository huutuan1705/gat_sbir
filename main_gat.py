import argparse
from gat.model import FGSBIR_GAT
from gat.train import train_model

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='GAT Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=8)
    parsers.add_argument('--root_dir', type=str, default='./../')
    
    parsers.add_argument('--use_kaiming_init', type=bool, default=True)
    
    parsers.add_argument('--batch_size', type=int, default=48)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.0001)
    parsers.add_argument('--epochs', type=int, default=300)
    
    args = parsers.parse_args()
    model = FGSBIR_GAT(visual_feature_dim=1024,
                       gat_out_per_head_dim=256,
                       gcn_label_feature_dim=1024,
                       n_gat_heads=4,
                       final_embedding_dim=512)
    train_model(model, args)
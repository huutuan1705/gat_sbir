import argparse
from gat.model import MIGG
from gat.train import train_model
from gat.utils import get_model_config

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
    
    if args.dataset_name == "ChairV2":
        num_classes = 19
    else:
        num_classes = 15
    config = get_model_config()
    model = MIGG(num_classes=num_classes, config=config, args=args)
    
    train_model(model, args, num_classes=num_classes)
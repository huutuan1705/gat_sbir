import torch
import torch.nn as nn
import torch.utils.data as data 
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from gat.datasets import MIGG_Dataset
from gat.losses import compute_migg_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = MIGG_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = MIGG_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

def evaluate_model(model, test_loader, all_label_indices, label_adj_matrix):
    with torch.no_grad():
        model.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []
        for _, data_batch in enumerate(tqdm(test_loader)):
            sketch_features_all = torch.FloatTensor().to(device)
            _, _, _, positive_feature, _, sketch_features = model(
                data_batch,
                all_label_indices,
                label_adj_matrix
            )
        
def train_one_epoch(model, train_loader, optimizer, all_label_indices, label_adj_matrix,
                    loss_weights_config):
    model.train()
    total_loss_epoch = 0
    all_individual_losses_epoch = {}
    
    for _, data_batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        true_labels_multihot = data_batch['labels'].to(device)
        
        search_embeddings, prediction_scores, gcn_features, positive_feature, negative_feature, sketch_features = model(
            data_batch,
            all_label_indices,
            label_adj_matrix
        )
        
        loss, _ = compute_migg_loss(
            positive_feature=positive_feature.to(device),
            negative_feature=negative_feature.to(device),
            sketch_feature=sketch_features.to(device),
            gcn_processed_label_features=gcn_features,
            prediction_scores=prediction_scores,
            true_labels_multihot=true_labels_multihot,
            model_parameters=model.parameters(),
        )
        
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
    
    avg_loss = total_loss_epoch / len(train_loader)
    return avg_loss 
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data 
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from gat.datasets import MIGG_Dataset
from gat.losses import compute_migg_loss
from gat.utils import get_label_adjacency_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args, num_classes):
    dataset_train = MIGG_Dataset(args, mode='train', num_classes=num_classes)
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = MIGG_Dataset(args, mode='test', num_classes=num_classes)
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

def evaluate_model(model, dataloader_test, all_label_indices, label_adj_matrix):
    with torch.no_grad():
        model.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []
        
        for idx, batch in enumerate(tqdm(dataloader_test)):
            sketch_features_all = torch.FloatTensor().to(device)
            for data_sketch in batch['sketch_imgs']:
                # print(data_sketch.shape) # (1, 25, 3, 299, 299)
                sketch_feature = model.sketch_linear(model.sketch_attention(
                    model.sketch_embedding_network(data_sketch.to(device))
                ))
                # print("sketch_feature.shape: ", sketch_feature.shape) #(25, 2048)
                sketch_features_all = torch.cat((sketch_features_all, sketch_feature.detach()))
            
            # print("sketch_feature_ALL.shape: ", sketch_features_all.shape) # (25, 2048)           
            sketch_array_tests.append(sketch_features_all.cpu())
            sketch_names.extend(batch['sketch_path'])
            
            if batch['positive_path'][0] not in image_names:
                positive_feature = model.linear(model.attention(
                    model.sample_embedding_network(batch['positive_img'].to(device))))
                image_array_tests = torch.cat((image_array_tests, positive_feature))
                image_names.extend(batch['positive_path'])
        
        # print("sketch_array_tests[0].shape", sketch_array_tests[0].shape) #(25, 2048)
        num_steps = len(sketch_array_tests[0])
        avererage_area = []
        avererage_area_percentile = []
                
        rank_all = torch.zeros(len(sketch_array_tests), num_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), num_steps)
                
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            mean_rank = []
            mean_rank_percentile = []
            sketch_name = sketch_names[i_batch]
            
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            sketch_features = sampled_batch
            
            for i_sketch in range(sampled_batch.shape[0]):
                # print("sketch_features[i_sketch].shape: ", sketch_features[i_sketch].shape)
                sketch_feature = sketch_features[i_sketch]
                target_distance = F.pairwise_distance(sketch_feature.to(device), image_array_tests[position_query].to(device))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0).to(device), image_array_tests.to(device))
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()

                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                        #1/(rank)
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
            
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
        
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        
        meanMA = np.mean(avererage_area_percentile)
        meanMB = np.mean(avererage_area)
        
        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB
        
def train_one_epoch(model, train_loader, optimizer, all_label_indices, label_adj_matrix):
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

def train_model(model, args):
    if args.dataset_name == "ChairV2":
        num_classes = 19
    else:
        num_classes = 15
        
    dataloader_train, dataloader_test = get_dataloader(args, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    all_label_indices = torch.arange(num_classes).to(device)
    label_adj_matrix = get_label_adjacency_matrix(
        dataloader_train, 
        num_classes, 
        device,
        threshold_type="dynamic",
        cooccurrence_threshold=0.01
    ).to(device)
    
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        
        avg_train_loss = train_one_epoch(
            model, dataloader_train, optimizer, all_label_indices, label_adj_matrix
        )
        
        top1_eval, top5_eval, top10_eval, meanA, meanB = evaluate_model(model, dataloader_test)
        if top5_eval > top5:
            top1, top5, top10 = top1_eval, top5_eval, top10_eval
            torch.save(model.state_dict(), "best_model.pth")
            
        print('Top 1 accuracy:  {:.4f}'.format(top1_eval))
        print('Top 5 accuracy:  {:.4f}'.format(top5_eval))
        print('Top 10 accuracy: {:.4f}'.format(top10_eval))
        print('Mean A         : {:.4f}'.format(meanA))
        print('Mean B         : {:.4f}'.format(meanB))
        print('Loss:            {:.4f}'.format(avg_train_loss))
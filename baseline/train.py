import torch
import torch.nn as nn
import torch.utils.data as data 

from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import StepLR
from baseline.datasets import FGSBIR_Dataset
from baseline.model import Basline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

def evaluate_model(model, args, dataloader_test):
    with torch.no_grad():
        model.eval()
        

def train_model(model, args):
    dataloader_train, dataloader_test = get_dataloader(args)
    
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
    
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        
        losses = []
        for _, batch_data in enumerate(tqdm(dataloader_train)):
            model.train()
            optimizer.zero_grad()
            
            sketch_feature, positive_feature, negative_feature = model(batch_data)
            loss = loss_fn(sketch_feature, positive_feature, negative_feature)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
        
        avg_loss = sum(losses) / len(losses)
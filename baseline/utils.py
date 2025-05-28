import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transform(type):
    if type == 'train':
        transform_list = [
            transforms.RandomResizedCrop(299, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    else: 
        transform_list = [
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    return transforms.Compose(transform_list)
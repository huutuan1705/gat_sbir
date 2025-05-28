import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transform(type):
    if type == 'train':
        transform_list = [
            transforms.Resize(299),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.25),
            transforms.RandomRotation(10),
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
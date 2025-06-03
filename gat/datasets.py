import os
import pickle
import torch

from torch.utils.data import Dataset
from random import randint
from PIL import Image

from baseline.utils import get_transform
from baseline.rasterize import rasterize_sketch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MIGG_Dataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        
        coordinate_path = os.path.join(args.root_dir, args.dataset_name, args.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(args.root_dir, args.dataset_name)
        with open(coordinate_path, 'rb') as f:
            self.coordinate = pickle.load(f)
            
        self.train_sketch = [x for x in self.coordinate if 'train' in x]
        self.test_sketch = [x for x in self.coordinate if 'test' in x]
        
        self.train_transform = get_transform('train')
        self.test_transform = get_transform('test')
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_sketch)

        return len(self.test_sketch)
        
        
    def __getitem__(self, item):
        sample = {}
        
        if self.mode == 'train':
            sketch_path = self.train_sketch[item]
            
            positive_sample = '_'.join(self.train_sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            
            posible_list = list(range(len(self.train_sketch)))
            posible_list.remove(item)
            
            negative_item = posible_list[randint(0, len(posible_list)-1)]
            negative_sample = '_'.join(self.train_sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')
            
            vector_x = self.coordinate[sketch_path]
            sketch_img = rasterize_sketch(vector_x)
import os
import pickle
import torch

from torch.utils.data import Dataset
from random import randint
from PIL import Image

from baseline.utils import get_transform
from baseline.rasterize import rasterize_sketch_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(Dataset):
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
        
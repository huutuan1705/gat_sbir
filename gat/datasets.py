import os
import pickle
import pandas as pd
import torch

from torch.utils.data import Dataset
from random import randint
from PIL import Image

from baseline.utils import get_transform
from baseline.rasterize import rasterize_sketch_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MIGG_Dataset(Dataset):
    def __init__(self, args, mode, num_classes):
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
        self.img_labels_df = pd.read_csv(args.annotations_file)
        self.num_classes = num_classes
                
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_sketch)

        return len(self.test_sketch)
        
        
    def __getitem__(self, item):
        labels = self.img_labels_df.iloc[item, -self.num_classes:].values
        labels = torch.tensor(labels.astype('float32'))
        
        if self.mode == 'train':
            positive_sample = '_'.join(self.train_sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            
            posible_list = list(range(len(self.train_sketch)))
            posible_list.remove(item)
            
            negative_item = posible_list[randint(0, len(posible_list)-1)]
            negative_sample = '_'.join(self.train_sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')
            
            positive_image = Image.open(positive_path).convert("RGB")
            negative_image = Image.open(negative_path).convert("RGB")
            positive_image = self.train_transform(positive_image)
            negative_image = self.train_transform(negative_image)
            
            sketch_path = self.train_sketch[item]
            vector_x = self.coordinate[sketch_path]
            list_sketch_imgs = rasterize_sketch_steps(vector_x)
            
            if self.args.on_fly:
                sketch_raw_imgs = [Image.fromarray(sk_img).convert("RGB") for sk_img in list_sketch_imgs]
                sketch_images = torch.stack([self.train_transform(sk_img) for sk_img in sketch_raw_imgs])
            else:
                sketch_images = self.train_transform(Image.fromarray(list_sketch_imgs[-1]).convert("RGB"))
                
            return {
                "positive_image": positive_image,
                "negative_image": negative_image,
                "sketch_images": sketch_images,
                "labels": labels
            }
        
        else:
            sketch_path = self.test_sketch[item] 
            vector_x = self.coordinate[sketch_path]
            
            list_sketch_imgs = rasterize_sketch_steps(vector_x)
            
            if self.args.on_fly:
                sketch_raw_imgs = [Image.fromarray(sk_img).convert("RGB") for sk_img in list_sketch_imgs]
                sketch_images = torch.stack([self.test_transform(sk_img) for sk_img in sketch_raw_imgs])
            else:
                sketch_images = self.test_transform(Image.fromarray(list_sketch_imgs[-1]).convert("RGB"))
                
            return {
                "positive_image": positive_image,
                "sketch_images": sketch_images,
                "labels": labels
            }
import torch
import torch.nn as nn
import torch.utils.data as data 
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from gat.datasets import MIGG_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

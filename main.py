import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import data_loader


device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print('device:', device)

if __name__ == '__main__':
    pass
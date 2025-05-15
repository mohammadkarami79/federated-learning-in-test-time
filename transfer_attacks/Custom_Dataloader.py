"""
Custom Dataloader module for transfer attacks
"""

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """
    A custom dataset that can be used for transfer attacks
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        
        if self.transform:
            data = self.transform(data)
            
        return data, label

# Add any other necessary classes or functions that might be imported from this module 
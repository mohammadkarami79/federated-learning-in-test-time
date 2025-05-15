"""
Data utilities for loading and partitioning datasets
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
from config import DATA_DIR

def get_dataloader(cfg, split="train"):
    """
    Unified data loader function based on configuration
    
    Args:
        cfg: Configuration object containing dataset settings
        split: Data split ('train' or 'test')
        
    Returns:
        DataLoader: Dataset loader
    """
    if cfg.DATASET_NAME == "CIFAR10":
        if split == "train":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
        ds = torchvision.datasets.CIFAR10(
            root=cfg.DATA_PATH, 
            train=(split=="train"),
            transform=transform, 
            download=True
        )
    elif cfg.DATASET_NAME == "MedMNIST":
        # Basic transform for MedMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # TODO: implement MedMNIST loader
        # This would require the medmnist package
        # Example implementation:
        # from medmnist import PneumoniaMNIST
        # ds = PneumoniaMNIST(root=cfg.DATA_PATH, split=split, transform=transform, download=True)
        raise NotImplementedError("MedMNIST support not yet implemented")
    else:
        raise ValueError(f"Unknown dataset {cfg.DATASET_NAME}")
    
    return torch.utils.data.DataLoader(
        ds, 
        batch_size=cfg.BATCH_SIZE,
        shuffle=(split=="train"),
        num_workers=2,
        pin_memory=True
    )

def get_dataset(
    dataset_name: str,
    train: bool = True,
    batch_size: int = 32,
    num_workers: int = 2,
    transform: Optional[transforms.Compose] = None
) -> DataLoader:
    """
    Get dataset loader
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist', etc.)
        train: Whether to get training set
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        transform: Optional transform to apply to the data
        
    Returns:
        DataLoader: Dataset loader
    """
    if transform is None:
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get dataset
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR),
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation and test sets
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training set
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation and test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Ratios must sum to 1"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_set, val_set, test_set

def get_data_info(dataset_name: str) -> dict:
    """
    Get dataset information
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        dict: Dataset information
    """
    if dataset_name == 'cifar10':
        return {
            'num_classes': 10,
            'input_channels': 3,
            'input_size': 32,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010)
        }
    elif dataset_name == 'mnist':
        return {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': 28,
            'mean': (0.1307,),
            'std': (0.3081,)
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_data_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """
    Get data transforms for a dataset
    
    Args:
        dataset_name: Name of the dataset
        train: Whether to get training transforms
        
    Returns:
        transforms.Compose: Composed transforms
    """
    if dataset_name == 'cifar10':
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    elif dataset_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_cifar10_data(batch_size=64, num_workers=2):
    """
    Get CIFAR-10 data loaders
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def create_data_loaders(cfg) -> Tuple[List[DataLoader], DataLoader]:
    """Create train and test data loaders.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (train_loaders, test_loader)
    """
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create non-IID partitions for training
    train_loaders = create_non_iid_loaders(
        dataset=train_dataset,
        n_clients=cfg.N_CLIENTS,
        alpha=cfg.DIRICHLET_ALPHA,
        batch_size=cfg.BATCH_SIZE
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    return train_loaders, test_loader

def create_non_iid_loaders(
    dataset: Dataset,
    n_clients: int,
    alpha: float,
    batch_size: int
) -> List[DataLoader]:
    """Create non-IID data partitions using Dirichlet distribution.
    
    Args:
        dataset: PyTorch dataset
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter
        batch_size: Batch size
        
    Returns:
        List of DataLoaders for each client
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
        
    n_classes = len(np.unique(labels))
    
    # Generate Dirichlet distribution
    client_dist = np.random.dirichlet([alpha] * n_clients, n_classes)
    
    # Partition indices
    client_idxs = [[] for _ in range(n_clients)]
    for k in range(n_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Get proportions for this class
        proportions = client_dist[k]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split indices
        splits = np.split(idx_k, proportions)
        for client_idx, split in enumerate(splits):
            client_idxs[client_idx].extend(split)
            
    # Create dataloaders
    loaders = []
    for idxs in client_idxs:
        client_dataset = Subset(dataset, idxs)
        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        loaders.append(loader)
        
    return loaders
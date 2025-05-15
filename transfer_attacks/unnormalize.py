"""
Unnormalize module for transfer attacks
"""

import torch

def unnormalize(tensor, mean, std):
    """
    Unnormalize a tensor that was normalized with the given mean and std
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Unnormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Add any other necessary functions that might be imported from this module 
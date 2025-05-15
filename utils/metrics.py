"""
Utility functions for computing metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.metrics import structural_similarity as ssim


def mse(y_pred, y):
    return F.mse_loss(y_pred, y)


def binary_accuracy(y_pred, y):
    y_pred = torch.round(torch.sigmoid(y_pred))  # round predictions to the closest integer
    correct = (y_pred == y).float()
    acc = correct.sum()
    return acc


def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum()
    return acc


def compute_metrics(clean_images, adv_images, purified_images, model, device):
    """
    Compute various metrics for evaluation
    
    Args:
        clean_images (torch.Tensor): Clean images
        adv_images (torch.Tensor): Adversarial images
        purified_images (torch.Tensor): Purified images
        model (nn.Module): Target model
        device (torch.device): Device to run on
        
    Returns:
        dict: Dictionary containing metrics
    """
    metrics = {}
    
    # Detach tensors
    clean_images = clean_images.detach()
    adv_images = adv_images.detach()
    purified_images = purified_images.detach()
    
    # Compute accuracy
    with torch.no_grad():
        clean_outputs = model(clean_images.to(device))
        adv_outputs = model(adv_images.to(device))
        purified_outputs = model(purified_images.to(device))
        
        clean_preds = clean_outputs.argmax(dim=1)
        adv_preds = adv_outputs.argmax(dim=1)
        purified_preds = purified_outputs.argmax(dim=1)
        
        metrics['clean_acc'] = (clean_preds == clean_preds).float().mean().item()
        metrics['adv_acc'] = (clean_preds == adv_preds).float().mean().item()
        metrics['defense_acc'] = (clean_preds == purified_preds).float().mean().item()
    
    # Compute MSE
    metrics['mse_clean_adv'] = F.mse_loss(clean_images, adv_images).item()
    metrics['mse_clean_purified'] = F.mse_loss(clean_images, purified_images).item()
    
    # Compute SSIM
    ssim_clean_adv = []
    ssim_clean_purified = []
    
    for i in range(len(clean_images)):
        clean_img = clean_images[i].cpu().numpy().transpose(1, 2, 0)
        adv_img = adv_images[i].cpu().numpy().transpose(1, 2, 0)
        purified_img = purified_images[i].cpu().numpy().transpose(1, 2, 0)
        
        ssim_clean_adv.append(ssim(clean_img, adv_img, multichannel=True))
        ssim_clean_purified.append(ssim(clean_img, purified_img, multichannel=True))
    
    metrics['ssim_clean_adv'] = np.mean(ssim_clean_adv)
    metrics['ssim_clean_purified'] = np.mean(ssim_clean_purified)
    
    return metrics


def calculate_metrics(model, dataloader, device):
    """
    Calculate model metrics on a dataset
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        device (torch.device): Device to run on
        
    Returns:
        dict: Dictionary containing metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }

"""
Utility functions for visualization
"""

import os
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import numpy as np
from pathlib import Path

def save_images(clean_images, adv_images, def_images, labels, output_dir):
    """
    Save images for visualization
    
    Args:
        clean_images (torch.Tensor): Original clean images
        adv_images (torch.Tensor): Adversarial images
        def_images (torch.Tensor): Defended images
        labels (torch.Tensor): Image labels
        output_dir (str): Directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original images
    vutils.save_image(clean_images, 
                     os.path.join(output_dir, 'original.png'),
                     normalize=True)
    
    # Save adversarial images
    vutils.save_image(adv_images,
                     os.path.join(output_dir, 'adversarial.png'),
                     normalize=True)
    
    # Save defended images
    vutils.save_image(def_images,
                     os.path.join(output_dir, 'defended.png'),
                     normalize=True)

def plot_results(results, save_path=None):
    """
    Plot experiment results
    
    Args:
        results (dict): Dictionary containing results
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(results['clean_acc'], label='Clean')
    plt.plot(results['adv_acc'], label='Adversarial')
    plt.plot(results['defense_acc'], label='Defense')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Comparison')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(results['clean_loss'], label='Clean')
    plt.plot(results['adv_loss'], label='Adversarial')
    plt.plot(results['defense_loss'], label='Defense')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_images_with_predictions(images, labels, preds, save_dir, prefix=''):
    """
    Save images with predictions
    
    Args:
        images (torch.Tensor): Input images
        labels (torch.Tensor): True labels
        preds (torch.Tensor): Predicted labels
        save_dir (str): Directory to save images
        prefix (str): Prefix for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (img, label, pred) in enumerate(zip(images, labels, preds)):
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f'True: {label}, Pred: {pred}')
        plt.axis('off')
        
        save_path = save_dir / f'{prefix}_{i}.png'
        plt.savefig(str(save_path))
        plt.close()

def plot_diffusion_process(images, save_path=None):
    """
    Plot diffusion process
    
    Args:
        images (list): List of images at different diffusion steps
        save_path (str, optional): Path to save the plot
    """
    n_steps = len(images)
    fig, axes = plt.subplots(1, n_steps, figsize=(4*n_steps, 4))
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'Step {i}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 
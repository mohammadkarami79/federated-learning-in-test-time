"""
Utility functions for pFedDef defense mechanism based on the original paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from typing import List, Dict, Tuple

def load_client_models(client_ids: List[int], model_name: str, dataset: str, device: torch.device, models_dir: str) -> List[nn.Module]:
    """
    Load client models for ensemble defense.
    
    Args:
        client_ids: List of client IDs to load models for
        model_name: Type of model to load (e.g., 'resnet18')
        dataset: Dataset name (e.g., 'cifar10')
        device: Device to load models on
        models_dir: Directory containing model weights
        
    Returns:
        List of loaded client models
    """
    from main import load_target_model  # Import here to avoid circular imports
    
    models = []
    for client_id in client_ids:
        # Load base model architecture
        model = load_target_model(model_name, dataset, device)
        
        # Load client-specific weights
        model_path = os.path.join(models_dir, f'client_{client_id}', f'{model_name}_{dataset}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded client {client_id} model from {model_path}")
        else:
            print(f"Warning: Client {client_id} model weights not found at {model_path}. Using global model.")
        
        models.append(model)
    
    return models

def robust_aggregation(predictions: torch.Tensor, method: str = 'median') -> torch.Tensor:
    """
    Aggregate predictions from multiple models using robust methods.
    
    Args:
        predictions: Tensor of shape [num_models, batch_size, num_classes]
        method: Aggregation method ('median', 'trimmed_mean', 'majority_vote')
        
    Returns:
        Aggregated predictions of shape [batch_size, num_classes]
    """
    if method == 'median':
        return torch.median(predictions, dim=0)[0]
    elif method == 'trimmed_mean':
        # Remove top and bottom 20% of predictions
        k = int(0.2 * predictions.shape[0])
        sorted_preds = torch.sort(predictions, dim=0)[0]
        trimmed = sorted_preds[k:-k]
        return torch.mean(trimmed, dim=0)
    elif method == 'majority_vote':
        # Convert to class predictions
        class_preds = torch.argmax(predictions, dim=2)
        # Get mode for each sample
        return torch.mode(class_preds, dim=0)[0]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def compute_client_confidence(model: nn.Module, val_loader: torch.utils.data.DataLoader, 
                            device: torch.device) -> float:
    """
    Compute confidence score for a client model based on validation performance.
    
    Args:
        model: Client model
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Confidence score between 0 and 1
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    return accuracy

def adaptive_weighting(models: List[nn.Module], val_loader: torch.utils.data.DataLoader,
                      device: torch.device) -> torch.Tensor:
    """
    Compute adaptive weights for ensemble based on validation performance.
    
    Args:
        models: List of client models
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Normalized weights for each model
    """
    confidences = torch.tensor([
        compute_client_confidence(model, val_loader, device)
        for model in models
    ], device=device)
    
    # Softmax to get normalized weights
    weights = F.softmax(confidences, dim=0)
    return weights

def pfeddef_predict(models: List[nn.Module], images: torch.Tensor,
                   val_loader: torch.utils.data.DataLoader, device: torch.device,
                   confidence_threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions using pFedDef ensemble with adaptive weighting and confidence filtering.
    
    Args:
        models: List of client models
        images: Input images [batch_size, channels, height, width]
        val_loader: Validation data loader for computing weights
        device: Device to run on
        confidence_threshold: Minimum confidence threshold for predictions
        
    Returns:
        Tuple of (predictions, confidences)
    """
    # Get predictions from each model
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, preds = probs.max(dim=1)
            all_predictions.append(preds)
            all_confidences.append(max_probs)
    
    # Stack predictions and confidences
    predictions = torch.stack(all_predictions, dim=1)  # [batch_size, num_models]
    confidences = torch.stack(all_confidences, dim=1)  # [batch_size, num_models]
    
    # Filter low confidence predictions
    mask = confidences >= confidence_threshold
    filtered_preds = predictions[mask]
    filtered_confs = confidences[mask]
    
    if filtered_preds.shape[0] > 0:
        # Weighted voting based on confidence
        weighted_votes = torch.zeros(
            (filtered_preds.shape[0], models[0].fc.out_features),
            device=device
        )
        
        for i in range(filtered_preds.shape[1]):
            votes = F.one_hot(filtered_preds[:, i], models[0].fc.out_features)
            weighted_votes += votes * filtered_confs[:, i].unsqueeze(1)
        
        final_preds = weighted_votes.argmax(dim=1)
    else:
        # Fallback to simple majority voting
        final_preds = predictions.mode(dim=1)[0]
    
    return final_preds, confidences.mean(dim=1)

def diffusion_ensemble_predict(models: List[nn.Module], diffusion_models: List[nn.Module], 
                              images: torch.Tensor, aggregation_method: str = 'median') -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Apply combined diffusion purification and pFedDef ensemble prediction.
    
    Args:
        models: List of client models
        diffusion_models: List of client diffusion models
        images: Adversarial images [batch_size, channels, height, width]
        aggregation_method: Method for aggregating predictions
        
    Returns:
        Tuple of (aggregated_prediction_tensor, prediction_labels, purified_images_list)
    """
    with torch.no_grad():
        # Get predictions from each model with purification
        all_predictions = []
        purified_images_list = []
        
        for model, diffusion_model in zip(models, diffusion_models):
            model.eval()
            diffusion_model.eval()
            
            # Purify the adversarial images
            purified_images = diffusion_model.purify(images)
            purified_images_list.append(purified_images)
            
            # Make predictions on purified images
            outputs = model(purified_images)
            all_predictions.append(outputs)
        
        # Stack predictions [num_models, batch_size, num_classes]
        all_predictions = torch.stack(all_predictions)
        
        # Aggregate predictions
        aggregated_preds = robust_aggregation(all_predictions, method=aggregation_method)
        
        # Get the predicted class
        pred_labels = torch.argmax(aggregated_preds, dim=1)
        
        return aggregated_preds, pred_labels, purified_images_list 
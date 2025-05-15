"""
Combined defense mechanism using DiffPure and pFedDef
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Tuple

from utils.model_manager import ModelManager
from diffusion.diffuser import UNet
from models.resnet import ResNet18
from defense.mae_detector import MAEDetector

class CombinedDefense:
    """
    Implements combined defense using DiffPure and pFedDef
    """
    def __init__(self, dataset: str = 'cifar10', client_id: int = 0):
        """
        Initialize combined defense
        
        Args:
            dataset (str): Dataset name
            client_id (int): Client ID for federated models
        """
        self.dataset = dataset
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Load models
        self.diffusion_model = self._load_diffusion_model()
        self.pfeddef_model = self._load_pfeddef_model()
        
    def _load_diffusion_model(self) -> nn.Module:
        """Load DiffPure model"""
        model = UNet(in_channels=3, hidden_channels=64)
        model_path = self.model_manager.get_latest_model('diffusion', self.dataset)
        
        try:
            model = self.model_manager.load_model(model, model_path)
            self.logger.info("Successfully loaded DiffPure model")
        except FileNotFoundError:
            self.logger.warning("No pre-trained DiffPure model found. Using untrained model.")
        
        return model.to(self.device)
        
    def _load_pfeddef_model(self) -> nn.Module:
        """Load pFedDef model"""
        model = ResNet18(num_classes=10)  # Assuming CIFAR-10
        model_path = self.model_manager.get_latest_model('pfeddef', self.dataset, self.client_id)
        
        try:
            model = self.model_manager.load_model(model, model_path)
            self.logger.info("Successfully loaded pFedDef model")
        except FileNotFoundError:
            self.logger.warning("No pre-trained pFedDef model found. Using untrained model.")
        
        return model.to(self.device)
        
    def purify_images(self, images: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """
        Purify images using DiffPure
        
        Args:
            images (torch.Tensor): Input images
            sigma (float): Noise level for diffusion
            
        Returns:
            torch.Tensor: Purified images
        """
        self.diffusion_model.eval()
        with torch.no_grad():
            # Add noise
            noise = torch.randn_like(images) * sigma
            noisy_images = images + noise
            
            # Generate timesteps
            batch_size = images.shape[0]
            t = torch.ones(batch_size, device=self.device)  # Full denoising
            
            # Predict and remove noise
            pred_noise = self.diffusion_model(noisy_images, t)
            purified_images = noisy_images - pred_noise
            
            return purified_images
            
    def defend(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply combined defense to images
        
        Args:
            images (torch.Tensor): Input images
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Purified images and predictions
        """
        # Move images to device
        images = images.to(self.device)
        
        # Step 1: Purify images using DiffPure
        purified_images = self.purify_images(images)
        
        # Step 2: Apply pFedDef defense
        self.pfeddef_model.eval()
        with torch.no_grad():
            predictions = self.pfeddef_model(purified_images)
            
        return purified_images, predictions
        
    def save_models(self) -> None:
        """Save both models"""
        # Save DiffPure model
        diffusion_path = self.model_manager.get_latest_model('diffusion', self.dataset)
        self.model_manager.save_model(self.diffusion_model, diffusion_path)
        
        # Save pFedDef model
        pfeddef_path = self.model_manager.get_latest_model('pfeddef', self.dataset, self.client_id)
        self.model_manager.save_model(self.pfeddef_model, pfeddef_path)
        
    def load_models(self) -> None:
        """Reload both models from disk"""
        self.diffusion_model = self._load_diffusion_model()
        self.pfeddef_model = self._load_pfeddef_model()

class CombinedClassifier(nn.Module):
    def __init__(self, diffuser, pfeddef_model, cfg):
        super().__init__()
        self.diffuser = diffuser
        self.pfeddef = pfeddef_model
        self.cfg = cfg
        self.disable_defense = False
        self.temperature = nn.Parameter(torch.ones(1) * 0.2)
        self.adv_training = True
        
        # Initialize MAE detector if enabled
        self.mae_detector = None
        if getattr(self.cfg, 'ENABLE_MAE_DETECTOR', False):
            self.mae_detector = MAEDetector(cfg)
            
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def forward(self, x: torch.Tensor, client_id: Optional[int] = None) -> torch.Tensor:
        if self.disable_defense:
            return self.pfeddef(x, client_id)
        
        # Use MAE detector if enabled
        if self.mae_detector is not None and not self.training:
            # Detect if inputs are adversarial
            is_adversarial = self.mae_detector.detect(x)
            
            # If any adversarial examples detected, apply defense only to those
            if is_adversarial.sum() > 0:
                self.logger.info(f"MAE Detector: Found {is_adversarial.sum().item()} adversarial examples")
                
                # Create a copy of the input for processing
                x_processed = x.clone()
                
                # Apply defense only to adversarial examples
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    # Process adversarial examples
                    adv_indices = torch.where(is_adversarial == 1)[0]
                    x_adv = x[adv_indices]
                    
                    if self.training:
                        adv_outputs = self._forward_train(x_adv, client_id)
                    else:
                        adv_outputs = self._forward_eval(x_adv, client_id)
                    
                    # Replace outputs for adversarial examples
                    for i, idx in enumerate(adv_indices):
                        x_processed[idx] = adv_outputs[i]
                    
                # Return the processed batch
                return self.pfeddef(x_processed, client_id)
            else:
                # No adversarial examples detected, use standard forward pass
                return self.pfeddef(x, client_id)
                
        # No MAE detector or in training mode - use standard defense
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # Different paths for training and evaluation
            if self.training:
                return self._forward_train(x, client_id)
            else:
                return self._forward_eval(x, client_id)
    
    def _forward_eval(self, x: torch.Tensor, client_id: Optional[int] = None) -> torch.Tensor:
        """Evaluation forward pass - no gradients or adversarial training"""
        # Multiple purification passes with stronger noise levels and more diverse approaches
        x_pur_list = []
        noise_levels = [0.5, 0.6, 0.7, 0.8]  # Add even stronger noise level
        
        # First pass - standard purification
        for sigma in noise_levels:
            x_pur = self.purify_images(x, sigma)
            x_pur_list.append(x_pur)
        
        # Second pass - reversed noise order with different step counts
        for sigma in reversed(noise_levels):
            # Use different step counts for diversity
            extra_steps = 2 + int(sigma * 10)  # More steps for higher sigma
            x_pur = self.purify_images(x, sigma, extra_steps=extra_steps)
            x_pur_list.append(x_pur)
            
        # Third pass - add random noise and purify again (more iterations)
        for _ in range(3):  # Increased from 2 to 3
            # Add different noise patterns
            noise_scale = 0.2 + 0.1 * torch.rand(1).item()
            noisy_x = x + torch.randn_like(x) * noise_scale
            noisy_x = torch.clamp(noisy_x, 0, 1)
            x_pur = self.purify_images(noisy_x, 0.5)
            x_pur_list.append(x_pur)
            
        # Fourth pass - apply stronger noise transformation
        blur_kernel_size = 3
        for _ in range(2):
            # Create blur kernel
            channels = x.shape[1]
            kernel = torch.ones(channels, 1, blur_kernel_size, blur_kernel_size) / (blur_kernel_size * blur_kernel_size)
            kernel = kernel.to(x.device)
            
            # Apply blur and then purify
            padded_x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
            blurred_x = torch.nn.functional.conv2d(
                padded_x,
                kernel,
                groups=channels
            )
            blurred_x = torch.clamp(blurred_x, 0, 1)
            x_pur = self.purify_images(blurred_x, 0.6)
            x_pur_list.append(x_pur)
        
        # Get predictions from all learners with stronger ensemble
        logits_list = []
        weights_list = []
        
        for learner_idx, learner in enumerate(self.pfeddef.learners):
            learner_logits = []
            for x_pur in x_pur_list:
                # Get predictions with stronger noise injection for robustness
                logits = learner(x_pur)
                
                # Add noise for robust decision boundaries (stronger)
                noise_scale = 0.2 * (1 + 0.5 * learner_idx / len(self.pfeddef.learners))
                logits += torch.randn_like(logits) * noise_scale
                
                # Make predictions more confident by sharpening
                temp_scale = 0.6 - 0.1 * learner_idx / len(self.pfeddef.learners)  # Lower temp = sharper
                logits = logits / temp_scale
                
                learner_logits.append(logits)
            
            # Robust averaging within each learner
            learner_stack = torch.stack(learner_logits)
            learner_probs = torch.softmax(learner_stack, dim=-1)
            
            # Calculate confidence for weighting
            conf = learner_probs.max(dim=-1)[0].mean()
            
            # Add entropy as diversity measure (lower entropy = more confident)
            entropy = -(learner_probs * torch.log(learner_probs + 1e-8)).sum(-1).mean()
            
            # Combine confidence and diversity
            weights_list.append(conf * (1.0 - 0.1 * entropy))
            
            # Average logits from this learner (more stable than averaging probabilities)
            avg_logits = learner_stack.mean(0)
            logits_list.append(avg_logits)
        
        # Weight by confidence
        weights = torch.tensor(weights_list, device=x.device)
        weights = torch.softmax(weights * 3.0, dim=0)  # Sharpen the confidence weighting (increased from 2.0)
        
        # Ensemble with confidence weighting
        weighted_logits = torch.stack([w * l for w, l in zip(weights, logits_list)])
        final_logits = weighted_logits.sum(0)
        
        # Apply final temperature scaling for more confident predictions
        return final_logits / 0.25  # Lower temperature = more confident predictions (decreased from 0.3)
            
    def _forward_train(self, x: torch.Tensor, client_id: Optional[int] = None) -> torch.Tensor:
        """Training forward pass - includes adversarial training"""
        # Multiple purification passes
        x_pur_list = []
        
        # First pass - standard purification with strong noise
        x_pur1 = self.purify_images(x, sigma=0.5)
        x_pur_list.append(x_pur1)
        
        # Second pass - add random noise and purify
        noise_scale = 0.2
        noisy_x = x + torch.randn_like(x) * noise_scale
        noisy_x = torch.clamp(noisy_x, 0, 1)
        x_pur2 = self.purify_images(noisy_x, sigma=0.3)
        x_pur_list.append(x_pur2)
        
        # Get predictions from all learners
        logits_list = []
        
        for learner in self.pfeddef.learners:
            learner_logits = []
            for x_pur in x_pur_list:
                # Apply gradient noise during training for better robustness
                if x_pur.requires_grad:
                    # Register hook to add gradient noise
                    def add_gradient_noise(grad):
                        noise_level = 0.01
                        return grad + torch.randn_like(grad) * noise_level
                    
                    x_pur.register_hook(add_gradient_noise)
                
                # Get predictions
                logits = learner(x_pur)
                learner_logits.append(logits)
            
            # Average logits from this learner (more stable than averaging probabilities)
            avg_logits = torch.stack(learner_logits).mean(0)
            logits_list.append(avg_logits)
        
        # Simple average across learners
        final_logits = torch.stack(logits_list).mean(0)
        
        return final_logits
        
    def purify_images(self, x: torch.Tensor, sigma: Optional[float] = None, extra_steps: int = 0) -> torch.Tensor:
        """Purify images using diffusion model"""
        # Default sigma from config if not specified
        if sigma is None:
            sigma = self.cfg.DIFFUSER_SIGMA
            
        # Get steps from config and add extra steps if specified
        steps = self.cfg.DIFFUSER_STEPS + extra_steps
        
        return self.diffuser.purify(x, steps=steps, sigma=sigma)
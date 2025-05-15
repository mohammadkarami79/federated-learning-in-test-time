"""
Model manager for handling model operations
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any
import logging

class ModelManager:
    """
    Manages model operations including saving and loading
    """
    def __init__(self):
        """Initialize model manager"""
        self.base_dir = Path(os.getcwd())
        self.models_dir = self.base_dir / "models"
        self.target_dir = self.models_dir / "target"
        self.diffusion_dir = self.models_dir / "diffusion"
        self.pfeddef_dir = self.models_dir / "pfeddef"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create directories if they don't exist
        for directory in [self.models_dir, self.target_dir, self.diffusion_dir, self.pfeddef_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def save_model(self, model: torch.nn.Module, path: Path) -> None:
        """
        Save model to specified path
        
        Args:
            model (nn.Module): Model to save
            path (Path): Path to save model to
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, model: torch.nn.Module, path: Path) -> torch.nn.Module:
        """
        Load model from specified path
        
        Args:
            model (nn.Module): Model to load weights into
            path (Path): Path to load model from
            
        Returns:
            nn.Module: Loaded model
        """
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")
            
        model.load_state_dict(torch.load(path))
        self.logger.info(f"Model loaded from {path}")
        return model
        
    def get_latest_model(self, model_type: str, dataset: str = 'cifar10', client_id: int = 0) -> Path:
        """
        Get path to latest model of specified type
        
        Args:
            model_type (str): Type of model ('target', 'diffusion', 'pfeddef', or 'combined')
            dataset (str): Dataset name
            client_id (int): Client ID for federated models
            
        Returns:
            Path: Path to latest model
        """
        if model_type == 'target':
            return self.target_dir / f'target_{dataset}_resnet18.pth'
        elif model_type == 'diffusion':
            return self.diffusion_dir / dataset / f'diffusion_{dataset}_sigma0.1_steps50.pt'
        elif model_type == 'pfeddef':
            return self.pfeddef_dir / f'client_{client_id}' / f'pfeddef_resnet18_pgd_combined.pth'
        elif model_type == 'combined':
            return self.models_dir / 'combined' / f'combined_defense_{dataset}_client_{client_id}.pth'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def clean_old_models(self, model_type: str = None) -> None:
        """
        Clean old model files
        
        Args:
            model_type (str, optional): Type of models to clean. If None, clean all.
        """
        dirs_to_clean = []
        if model_type is None:
            dirs_to_clean = [self.target_dir, self.diffusion_dir, self.pfeddef_dir]
        elif model_type == 'target':
            dirs_to_clean = [self.target_dir]
        elif model_type == 'diffusion':
            dirs_to_clean = [self.diffusion_dir]
        elif model_type == 'pfeddef':
            dirs_to_clean = [self.pfeddef_dir]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        for directory in dirs_to_clean:
            if directory.exists():
                archive_dir = directory / 'archive'
                archive_dir.mkdir(parents=True, exist_ok=True)
                
                for file in directory.glob('*.pth'):
                    if file.name != self.get_latest_model(model_type).name:
                        new_path = archive_dir / file.name
                        file.rename(new_path)
                        self.logger.info(f"Moved {file} to {new_path}")
                        
    def get_model_info(self, model_type: str, dataset: str = 'cifar10') -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model_type (str): Type of model
            dataset (str): Dataset name
            
        Returns:
            dict: Model information
        """
        model_path = self.get_latest_model(model_type, dataset)
        
        info = {
            'type': model_type,
            'dataset': dataset,
            'path': model_path,
            'exists': model_path.exists()
        }
        
        if info['exists']:
            info['size'] = model_path.stat().st_size
            info['modified'] = model_path.stat().st_mtime
            
        return info 
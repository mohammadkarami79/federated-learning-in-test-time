"""
Script to migrate existing models to the new standardized structure
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

class ModelMigrator:
    """Handles migration of existing models to new structure"""
    
    def __init__(self, base_dir: str):
        """
        Initialize ModelMigrator
        
        Args:
            base_dir: Base directory containing old and new model structures
        """
        self.base_dir = Path(base_dir)
        self.old_weights_dir = self.base_dir / 'weights'
        self.old_models_dir = self.base_dir / 'models'
        self.new_models_dir = self.base_dir / 'models'
        
        # Create new directory structure
        self.diffusion_dir = self.new_models_dir / 'diffusion'
        self.target_dir = self.new_models_dir / 'target'
        self.pfeddef_dir = self.new_models_dir / 'pfeddef'
        self.client_weights_dir = self.new_models_dir / 'client_weights'
        
        for dir_path in [self.diffusion_dir, self.target_dir, self.pfeddef_dir, self.client_weights_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def find_old_models(self) -> Dict[str, List[Path]]:
        """
        Find all existing model files in old locations
        
        Returns:
            Dict mapping model types to lists of model paths
        """
        models = {
            'diffusion': [],
            'target': [],
            'pfeddef': [],
            'client': []
        }
        
        # Search in old weights directory
        if self.old_weights_dir.exists():
            # Search in diffpure directory
            diffpure_dir = self.old_weights_dir / 'cifar10' / 'diffpure'
            if diffpure_dir.exists():
                # Add diffusion model
                diffusion_model = diffpure_dir / 'diffusion_model.pth'
                if diffusion_model.exists():
                    models['diffusion'].append(diffusion_model)
                
                # Add client models
                for client_dir in diffpure_dir.glob('client_*'):
                    if client_dir.is_dir():
                        client_model = client_dir / 'model.pth'
                        if client_model.exists():
                            models['client'].append(client_model)
            
            # Search in pFedDef directory
            pfeddef_dir = self.old_weights_dir / 'cifar10' / 'pFedDef'
            if pfeddef_dir.exists():
                for model_file in pfeddef_dir.rglob('*.pth'):
                    if 'pfeddef' in model_file.name.lower():
                        models['pfeddef'].append(model_file)
        
        # Search in old models directory
        if self.old_models_dir.exists():
            # Add main models
            for model_file in self.old_models_dir.glob('*.pth'):
                if 'diffusion' in model_file.name.lower():
                    models['diffusion'].append(model_file)
                elif 'target' in model_file.name.lower():
                    models['target'].append(model_file)
                elif 'pfeddef' in model_file.name.lower():
                    models['pfeddef'].append(model_file)
            
            # Add client models
            for client_dir in self.old_models_dir.glob('client_*'):
                if client_dir.is_dir():
                    client_model = client_dir / 'model.pth'
                    if client_model.exists():
                        models['client'].append(client_model)
        
        return models
    
    def determine_model_info(self, model_path: Path) -> Dict[str, str]:
        """
        Determine model information from filename and path
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dict containing model information
        """
        info = {
            'dataset': 'unknown',
            'model_name': 'unknown',
            'attack_type': None,
            'defense_type': None,
            'client_id': None
        }
        
        # Determine dataset from path
        if 'cifar10' in str(model_path):
            info['dataset'] = 'cifar10'
        elif 'mnist' in str(model_path):
            info['dataset'] = 'mnist'
        
        # Determine model name
        if 'resnet' in str(model_path).lower():
            info['model_name'] = 'resnet18'
        elif 'vgg' in str(model_path).lower():
            info['model_name'] = 'vgg16'
        elif 'ddpm' in str(model_path).lower():
            info['model_name'] = 'ddpm'
        
        # Determine attack type
        if 'fgsm' in str(model_path).lower():
            info['attack_type'] = 'fgsm'
        elif 'pgd' in str(model_path).lower():
            info['attack_type'] = 'pgd'
        
        # Determine defense type
        if 'diffusion' in str(model_path).lower():
            info['defense_type'] = 'diffusion'
        elif 'pfeddef' in str(model_path).lower():
            info['defense_type'] = 'pfeddef'
        elif 'combined' in str(model_path).lower():
            info['defense_type'] = 'combined'
        
        # Determine client ID
        for part in model_path.parts:
            if part.startswith('client_'):
                try:
                    info['client_id'] = int(part.split('_')[1])
                except (ValueError, IndexError):
                    pass
        
        return info
    
    def get_new_path(self, model_type: str, info: Dict[str, str]) -> Path:
        """
        Get new standardized path for model
        
        Args:
            model_type: Type of model
            info: Model information dictionary
            
        Returns:
            New path for model
        """
        # Base filename components
        components = [info['dataset'], info['model_name']]
        
        # Add attack type if specified
        if info['attack_type']:
            components.append(f"attack_{info['attack_type']}")
            
        # Add defense type if specified
        if info['defense_type']:
            components.append(f"defense_{info['defense_type']}")
            
        # Create filename
        filename = "_".join(components) + ".pth"
        
        # Return appropriate directory path
        if model_type == 'diffusion':
            return self.diffusion_dir / filename
        elif model_type == 'target':
            return self.target_dir / filename
        elif model_type == 'pfeddef':
            return self.pfeddef_dir / filename
        elif model_type == 'client':
            if info['client_id'] is not None:
                client_dir = self.client_weights_dir / f"client_{info['client_id']}"
                client_dir.mkdir(exist_ok=True)
                return client_dir / filename
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def migrate_models(self) -> None:
        """Migrate all found models to new structure"""
        # Find existing models
        models = self.find_old_models()
        
        # Migrate each model
        for model_type, model_paths in models.items():
            for old_path in model_paths:
                # Determine model information
                info = self.determine_model_info(old_path)
                
                # Get new path
                new_path = self.get_new_path(model_type, info)
                
                # Create backup of old model
                backup_path = old_path.with_suffix('.pth.bak')
                shutil.copy2(old_path, backup_path)
                
                # Copy to new location
                try:
                    shutil.copy2(old_path, new_path)
                    logging.info(f"Migrated {old_path} to {new_path}")
                except Exception as e:
                    logging.error(f"Failed to migrate {old_path}: {str(e)}")

def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize migrator
    migrator = ModelMigrator('D:/papers/Team/pFedDef_v1_kaggle')
    
    # Run migration
    logging.info("Starting model migration...")
    migrator.migrate_models()
    logging.info("Model migration completed")

if __name__ == '__main__':
    main() 
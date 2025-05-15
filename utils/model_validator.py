"""
Model validator for checking and fixing model structure
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Validates and fixes model structure
    """
    def __init__(self):
        """Initialize model validator"""
        self.base_dir = Path("D:/papers/Team/pFedDef_v1_kaggle")
        self.models_dir = self.base_dir / "models"
        self.target_dir = self.models_dir / "target"
        self.diffusion_dir = self.models_dir / "diffusion"
        self.pfeddef_dir = self.models_dir / "pfeddef"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.model_manager = ModelManager()
    
    def validate_structure(self):
        """Validate model directory structure"""
        # Check main directories
        if not self.models_dir.exists():
            logger.info(f"Creating models directory at {self.models_dir}")
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
        # Check target model directory
        if not self.target_dir.exists():
            logger.info(f"Creating target model directory at {self.target_dir}")
            self.target_dir.mkdir(parents=True, exist_ok=True)
            
        # Check diffusion model directory
        if not self.diffusion_dir.exists():
            logger.info(f"Creating diffusion model directory at {self.diffusion_dir}")
            self.diffusion_dir.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories for diffusion models
        diffusion_general = self.diffusion_dir / "general"
        diffusion_cifar10 = diffusion_general / "cifar10"
        if not diffusion_cifar10.exists():
            logger.info(f"Creating CIFAR-10 diffusion model directory at {diffusion_cifar10}")
            diffusion_cifar10.mkdir(parents=True, exist_ok=True)
            
        # Check pFedDef model directory
        if not self.pfeddef_dir.exists():
            logger.info(f"Creating pFedDef model directory at {self.pfeddef_dir}")
            self.pfeddef_dir.mkdir(parents=True, exist_ok=True)
            
        # Create client directories
        client_dir = self.pfeddef_dir / "client_0"
        if not client_dir.exists():
            logger.info(f"Creating client directory at {client_dir}")
            client_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_old_models(self):
        """Check for and move old model files to archive."""
        archive_dir = self.base_dir / "archive" / "old_models"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Check each directory for old model files
        for directory in [self.target_dir, self.diffusion_dir, self.pfeddef_dir]:
            if directory.exists():
                for file in directory.glob("*.pth"):
                    try:
                        file.rename(archive_dir / file.name)
                        logger.info(f"Moved old model to archive: {file}")
                    except Exception as e:
                        logger.error(f"Error moving file {file}: {str(e)}")
                
                for file in directory.glob("*.pt"):
                    try:
                        file.rename(archive_dir / file.name)
                        logger.info(f"Moved old model to archive: {file}")
                    except Exception as e:
                        logger.error(f"Error moving file {file}: {str(e)}")
    
    def fix_structure(self):
        """Fix any issues in model structure"""
        # Create necessary directories
        self.validate_structure()
        
        # Check for required model files
        required_files = {
            'target': self.target_dir / 'target_cifar10_resnet18.pth',
            'diffusion': self.diffusion_dir / 'general' / 'cifar10' / 'diffusion_cifar10_sigma0.1_steps50.pt',
            'pfeddef': self.pfeddef_dir / 'client_0' / 'pfeddef_resnet18_pgd_combined.pth'
        }
        
        for model_type, file_path in required_files.items():
            if not file_path.exists():
                logger.warning(f"Warning: Missing {model_type} model file at {file_path}")
                logger.warning(f"Model will be trained and saved during execution")
    
    def migrate_existing_models(self) -> None:
        """
        Migrate existing models to the new structure
        """
        # Migrate diffusion models
        diffusion_dir = self.models_dir / 'diffusion'
        if diffusion_dir.exists():
            for model_file in diffusion_dir.glob('*.pth'):
                # Try to extract dataset and client info from filename
                filename = model_file.name
                if 'cifar10' in filename:
                    # Determine if it's a general or client model
                    if 'client' in filename:
                        # Extract client ID
                        client_id = int(filename.split('client')[1].split('_')[0])
                        if client_id == 0:  # Only migrate client 0 models
                            new_path = self.model_manager.get_diffusion_path('cifar10', client_id)
                            # Create backup and move
                            backup_path = self.models_dir / 'backups' / model_file.name
                            backup_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(model_file, backup_path)
                            shutil.move(model_file, new_path)
                            logger.info(f"Migrated {model_file} to {new_path}")
                    else:
                        new_path = self.model_manager.get_diffusion_path('cifar10')
                        # Create backup and move
                        backup_path = self.models_dir / 'backups' / model_file.name
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(model_file, backup_path)
                        shutil.move(model_file, new_path)
                        logger.info(f"Migrated {model_file} to {new_path}")
    
    def validate_and_fix(self) -> None:
        """
        Main function to validate and fix the model structure
        """
        logger.info("Starting model validation and fix")
        
        # Check current structure
        self.validate_structure()
        
        # Fix structure if needed
        self.fix_structure()
        self.migrate_existing_models()
        logger.info("Model structure fixed")
        
        # Final validation
        self.validate_structure()
        if any(issues.values()):
            logger.warning("Some issues remain after fix:")
            for category, problems in issues.items():
                if problems:
                    logger.warning(f"{category}:")
                    for problem in problems:
                        logger.warning(f"  - {problem}")
        else:
            logger.info("All issues resolved successfully")

if __name__ == "__main__":
    validator = ModelValidator()
    validator.validate_and_fix() 
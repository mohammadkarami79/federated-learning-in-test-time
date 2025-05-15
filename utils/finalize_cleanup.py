"""
Script to finalize model organization cleanup
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict

class ModelFinalization:
    """Handles final cleanup and organization of model structure"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / 'models'
        
        # Define standard directories
        self.standard_dirs = {
            'diffusion': self.models_dir / 'diffusion',
            'target': self.models_dir / 'target',
            'pfeddef': self.models_dir / 'pfeddef',
            'client_weights': self.models_dir / 'client_weights'
        }
        
        # Create backup directory
        self.backup_dir = self.models_dir / 'backups'
        self.backup_dir.mkdir(exist_ok=True)
    
    def organize_backups(self) -> None:
        """Move all backup files to a dedicated backup directory"""
        logging.info("Organizing backup files...")
        
        # Find and move all .bak files
        for file in self.models_dir.rglob('*.bak'):
            if self.backup_dir in file.parents:
                continue
                
            backup_path = self.backup_dir / file.name
            if not backup_path.exists():
                shutil.move(str(file), str(backup_path))
                logging.info(f"Moved backup file {file} to {backup_path}")
    
    def organize_pfeddef_models(self) -> None:
        """Organize models in pfeddef directory"""
        logging.info("Organizing PFedDef models...")
        
        pfeddef_dir = self.standard_dirs['pfeddef']
        if not pfeddef_dir.exists():
            return
            
        # Organize target models
        target_dir = pfeddef_dir / 'target_models'
        target_dir.mkdir(exist_ok=True)
        
        for file in pfeddef_dir.glob('*target*.pth'):
            if target_dir not in file.parents:
                target_path = target_dir / file.name
                if not target_path.exists():
                    shutil.move(str(file), str(target_path))
                    logging.info(f"Moved target model {file} to {target_path}")
        
        # Organize transfer models
        transfer_dir = pfeddef_dir / 'transfer_models'
        transfer_dir.mkdir(exist_ok=True)
        
        for file in pfeddef_dir.glob('*transfer*.pth'):
            if transfer_dir not in file.parents:
                transfer_path = transfer_dir / file.name
                if not transfer_path.exists():
                    shutil.move(str(file), str(transfer_path))
                    logging.info(f"Moved transfer model {file} to {transfer_path}")
    
    def cleanup_empty_dirs(self) -> None:
        """Remove empty directories"""
        logging.info("Cleaning up empty directories...")
        
        for root, dirs, files in os.walk(self.models_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        logging.info(f"Removed empty directory {dir_path}")
                except Exception as e:
                    logging.warning(f"Could not process directory {dir_path}: {str(e)}")
    
    def standardize_remaining_names(self) -> None:
        """Standardize names of any remaining non-standard files"""
        logging.info("Standardizing remaining filenames...")
        
        # Handle target models
        target_dir = self.standard_dirs['target']
        if target_dir.exists():
            for file in target_dir.glob('*.pth'):
                if 'target' not in file.stem:
                    new_name = f"cifar10_resnet18_target.pth"
                    new_path = file.parent / new_name
                    if not new_path.exists():
                        shutil.move(str(file), str(new_path))
                        logging.info(f"Renamed {file} to {new_path}")
        
        # Handle pfeddef models
        pfeddef_dir = self.standard_dirs['pfeddef']
        if pfeddef_dir.exists():
            for subdir in ['target_models', 'transfer_models']:
                subdir_path = pfeddef_dir / subdir
                if subdir_path.exists():
                    for file in subdir_path.glob('*.pth'):
                        if 'unknown' in file.stem:
                            model_type = 'target' if 'target' in subdir else 'transfer'
                            new_name = f"cifar10_resnet18_{model_type}.pth"
                            new_path = file.parent / new_name
                            if not new_path.exists():
                                shutil.move(str(file), str(new_path))
                                logging.info(f"Renamed {file} to {new_path}")
    
    def finalize(self) -> None:
        """Run all finalization operations"""
        logging.info("Starting model organization finalization...")
        
        self.organize_backups()
        self.organize_pfeddef_models()
        self.standardize_remaining_names()
        self.cleanup_empty_dirs()
        
        logging.info("Model organization finalization completed")

def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run finalization
    finalizer = ModelFinalization('D:/papers/Team/pFedDef_v1_kaggle')
    finalizer.finalize()

if __name__ == '__main__':
    main() 
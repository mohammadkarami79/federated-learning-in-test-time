"""
Script to clean up and standardize model organization
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List

class ModelCleanup:
    """Handles cleanup and standardization of model organization"""
    
    def __init__(self, base_dir: str):
        """
        Initialize ModelCleanup
        
        Args:
            base_dir: Base directory containing model structures
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / 'models'
        
        # Define standard directories
        self.standard_dirs = {
            'diffusion': self.models_dir / 'diffusion',
            'target': self.models_dir / 'target',
            'pfeddef': self.models_dir / 'pfeddef',
            'client_weights': self.models_dir / 'client_weights'
        }
    
    def move_root_models(self) -> None:
        """Move root-level model files to their respective directories"""
        for file in self.models_dir.glob('*.pth'):
            if 'diffusion' in file.name.lower():
                target_dir = self.standard_dirs['diffusion']
            elif 'target' in file.name.lower():
                target_dir = self.standard_dirs['target']
            elif 'pfeddef' in file.name.lower():
                target_dir = self.standard_dirs['pfeddef']
            else:
                continue
                
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / file.name
            
            if not target_path.exists():
                shutil.move(str(file), str(target_path))
                logging.info(f"Moved {file} to {target_path}")
            else:
                logging.warning(f"Target file {target_path} already exists, skipping {file}")
    
    def consolidate_client_dirs(self) -> None:
        """Consolidate client directories into client_weights"""
        client_weights_dir = self.standard_dirs['client_weights']
        client_weights_dir.mkdir(exist_ok=True)
        
        # Move client directories
        for client_dir in self.models_dir.glob('client_*'):
            if client_dir.is_dir() and client_dir != client_weights_dir:
                client_id = client_dir.name.split('_')[1]
                new_client_dir = client_weights_dir / f"client_{client_id}"
                
                if not new_client_dir.exists():
                    shutil.move(str(client_dir), str(new_client_dir))
                    logging.info(f"Moved {client_dir} to {new_client_dir}")
                else:
                    # Merge contents if target exists
                    for item in client_dir.iterdir():
                        target_path = new_client_dir / item.name
                        if not target_path.exists():
                            shutil.move(str(item), str(target_path))
                            logging.info(f"Moved {item} to {target_path}")
                    client_dir.rmdir()
                    logging.info(f"Merged contents of {client_dir} into {new_client_dir}")
    
    def remove_redundant_dirs(self) -> None:
        """Remove redundant directories"""
        redundant_dirs = [
            self.models_dir / 'diffusion_models'
        ]
        
        for dir_path in redundant_dirs:
            if dir_path.exists():
                # Move any files to standard directory first
                if dir_path.name == 'diffusion_models':
                    target_dir = self.standard_dirs['diffusion']
                    for file in dir_path.glob('*.pth'):
                        target_path = target_dir / file.name
                        if not target_path.exists():
                            shutil.move(str(file), str(target_path))
                            logging.info(f"Moved {file} to {target_path}")
                
                # Remove the directory
                dir_path.rmdir()
                logging.info(f"Removed redundant directory {dir_path}")
    
    def standardize_filenames(self) -> None:
        """Standardize model filenames across all directories"""
        for dir_type, dir_path in self.standard_dirs.items():
            if not dir_path.exists():
                continue
                
            for file in dir_path.glob('*.pth'):
                # Skip backup files
                if file.suffix == '.bak':
                    continue
                    
                # Get file info
                name_parts = file.stem.split('_')
                
                # Standardize based on directory type
                if dir_type == 'diffusion':
                    if 'unknown' in name_parts:
                        new_name = f"cifar10_ddpm_defense_diffusion.pth"
                    elif 'client' in name_parts:
                        new_name = f"cifar10_ddpm_client_{name_parts[2]}_sigma{name_parts[4]}_steps{name_parts[6]}.pth"
                    else:
                        continue
                elif dir_type == 'client_weights':
                    if len(name_parts) == 2 and name_parts[0] == 'client':
                        new_name = f"cifar10_resnet18_client_{name_parts[1]}.pth"
                    else:
                        continue
                else:
                    continue
                
                new_path = file.parent / new_name
                if not new_path.exists():
                    shutil.move(str(file), str(new_path))
                    logging.info(f"Renamed {file} to {new_path}")
    
    def cleanup(self) -> None:
        """Run all cleanup operations"""
        logging.info("Starting model cleanup...")
        
        self.move_root_models()
        self.consolidate_client_dirs()
        self.remove_redundant_dirs()
        self.standardize_filenames()
        
        logging.info("Model cleanup completed")

def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize cleanup
    cleanup = ModelCleanup('D:/papers/Team/pFedDef_v1_kaggle')
    
    # Run cleanup
    cleanup.cleanup()

if __name__ == '__main__':
    main() 
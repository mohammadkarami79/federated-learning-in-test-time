"""
Defense utilities for combining different defense strategies
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config_fixed import get_debug_config
from transfer_attacks.attacks import PGD, FGSM

class CombinedDefense:
    """
    Combined defense using diffusion and pFedDef
    """
    def __init__(self, diffusion_config, pfeddef_config, combined_config):
        self.diffusion_config = diffusion_config
        self.pfeddef_config = pfeddef_config
        self.combined_config = combined_config
        self.device = get_debug_config()['device']
        self.target_model = None  # Will be set during evaluation
        
    def evaluate_defense(self, test_loader, attack_type='pgd'):
        """
        Evaluate the combined defense against attacks
        
        Args:
            test_loader (DataLoader): Test data loader
            attack_type (str): Type of attack to evaluate against
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Load models first
        from utils.model_utils import get_target_model, get_diffusion_model, get_pfeddef_model
        
        self.target_model = get_target_model(
            name=get_debug_config()['model']['target']['name'],
            pretrained=False
        )
        self.target_model.load_state_dict(torch.load(get_debug_config()['model']['target']['checkpoint_path']))
        self.target_model.to(self.device)
        
        # Initialize attack with target model
        attack = PGD(
            model=self.target_model,
            epsilon=get_debug_config()['attack']['epsilon'],
            alpha=get_debug_config()['attack']['alpha'],
            steps=get_debug_config()['attack']['steps'],
            random_start=get_debug_config()['attack']['random_start']
        )
        
        diffusion_model = get_diffusion_model(
            in_channels=get_debug_config()['dataset']['input_shape'][0],
            hidden_channels=get_debug_config()['model']['diffusion']['hidden_channels'],
            num_blocks=get_debug_config()['model']['diffusion']['num_blocks']
        )
        diffusion_model.load_state_dict(torch.load(get_debug_config()['model']['diffusion']['checkpoint_path']))
        diffusion_model.to(self.device)
        
        pfeddef_model = get_pfeddef_model(
            name=get_debug_config()['model']['pfeddef']['name'],
            target_model=self.target_model,
            diffusion_model=diffusion_model
        )
        pfeddef_model.load_state_dict(torch.load(get_debug_config()['model']['pfeddef']['checkpoint_path']))
        pfeddef_model.to(self.device)
        
        # Set models to eval mode
        self.target_model.eval()
        diffusion_model.eval()
        pfeddef_model.eval()
        
        criterion = nn.CrossEntropyLoss()
        metrics = {
            'clean_accuracy': 0.0,
            'attack_accuracy': 0.0,
            'defense_accuracy': 0.0
        }
        
        with torch.no_grad():
            # Evaluate clean accuracy
            clean_correct = 0
            clean_total = 0
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.target_model(data)
                pred = output.argmax(dim=1)
                clean_correct += pred.eq(target).sum().item()
                clean_total += target.size(0)
            metrics['clean_accuracy'] = 100. * clean_correct / clean_total
            
            # Evaluate attack accuracy
            attack_correct = 0
            attack_total = 0
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                adv_data = attack.attack(data, target)
                output = self.target_model(adv_data)
                pred = output.argmax(dim=1)
                attack_correct += pred.eq(target).sum().item()
                attack_total += target.size(0)
            metrics['attack_accuracy'] = 100. * attack_correct / attack_total
            
            # Evaluate defense accuracy
            defense_correct = 0
            defense_total = 0
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                adv_data = attack.attack(data, target)
                
                # Apply diffusion
                clean_data = diffusion_model.purify(
                    adv_data,
                    steps=self.diffusion_config['n_steps'],
                    sigma=self.diffusion_config['t_star']
                )
                
                # Apply pFedDef
                output = pfeddef_model(clean_data)
                pred = output.argmax(dim=1)
                defense_correct += pred.eq(target).sum().item()
                defense_total += target.size(0)
            metrics['defense_accuracy'] = 100. * defense_correct / defense_total
        
        return metrics
    
    def run(self, target_model, diffusion_model, pfeddef_model, test_loader, device):
        """
        Run the combined defense
        
        Args:
            target_model (nn.Module): Target model
            diffusion_model (nn.Module): Diffusion model
            pfeddef_model (nn.Module): pFedDef model
            test_loader (DataLoader): Test data loader
            device (str): Device to use
        """
        target_model.eval()
        diffusion_model.eval()
        pfeddef_model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # Apply diffusion
                clean_data = diffusion_model.purify(
                    data,
                    steps=self.diffusion_config['n_steps'],
                    sigma=self.diffusion_config['t_star']
                )
                
                # Apply pFedDef
                output = pfeddef_model(clean_data)
                
                # Calculate metrics
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)
        
        # Print results
        avg_loss = total_loss / total_samples
        accuracy = 100. * total_correct / total_samples
        print(f"\nCombined Defense Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy 
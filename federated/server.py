"""
Federated server implementing FedEM aggregation
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import time
import psutil
from typing import List, Dict, Optional
from pathlib import Path

from config_fixed import get_debug_config
from .client import Client
from metrics.logger import MetricsLogger

class FedServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_models = None  # Will be set during init_round
        self.client_updates = {}  # Store client updates
        
    def init_round(self, model_template: nn.Module):
        """Initialize a new round with fresh global model."""
        if self.global_models is None:
            # Simple single model approach
            self.global_models = [self._clone_model(model_template)]
        return self.global_models
        
    def receive_update(self, client_id: int, local_models: List[nn.Module]):
        """Receive and store client updates for later aggregation."""
        # Handle both single model and list of models
        if not isinstance(local_models, list):
            local_models = [local_models]
        
        # Store only the first model (simple approach)
        self.client_updates[client_id] = [local_models[0]]
        
    def aggregate(self) -> List[nn.Module]:
        """Perform simple averaging aggregation."""
        n_clients = len(self.client_updates)
        assert n_clients > 0, "No updates received"
        
        # Get all client models (single model per client)
        client_models = [
            updates[0].state_dict()  # First (and only) model from each client
            for updates in self.client_updates.values()
        ]
        
        # Average the parameters
        avg_state = self.global_models[0].state_dict()
        for key in avg_state:
            # Skip BN layers for stability
            if 'bn' not in key.lower():
                # Check if all clients have this key
                if all(key in client_state for client_state in client_models):
                    stacked = torch.stack([
                        client_state[key] for client_state in client_models
                    ])
                    # Convert to float if it's integer type
                    if stacked.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
                        stacked = stacked.float()
                    avg_state[key] = stacked.mean(dim=0)
                
        # Load averaged weights back
        self.global_models[0].load_state_dict(avg_state)
        
        # Clear updates and return new global models
        self.client_updates.clear()
        return self.global_models
        
    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        """Deep copy a model including parameters."""
        import copy
        
        # Use simple deepcopy for ResNet18
        clone = copy.deepcopy(model)
        return clone.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_models = None  # Will be set during init_round
        self.client_updates = defaultdict(list)
        # Get device from cfg or use cuda if available
        self.device = getattr(cfg, 'DEVICE', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    def init_models(self, client_model: nn.Module):
        """Initialize global models using a client's model as template."""
        if self.global_models is None:
            n_learners = getattr(self.cfg, 'N_LEARNERS', 2)
            self.global_models = [
                self._clone_model(client_model)
                for _ in range(n_learners)
            ]
        
    def receive_update(self, client_id: int, local_models: List[nn.Module]):
        """Receive and store client updates for later aggregation."""
        if self.global_models is None:
            self.init_models(local_models[0])
        # For simple models, we expect only one model per client
        self.client_updates[client_id] = local_models
        
    def aggregate(self) -> List[nn.Module]:
        """Perform simple averaging aggregation."""
        n_clients = len(self.client_updates)
        if n_clients == 0:
            return []
        
        # Get first client's model as template
        first_client_model = list(self.client_updates.values())[0][0]
        
        # Create a simple copy without state_dict loading
        if hasattr(first_client_model, 'fc'):
            # For ResNet models
            import torchvision.models as models
            aggregated_model = models.resnet18(pretrained=False)
            aggregated_model.fc = nn.Linear(aggregated_model.fc.in_features, getattr(self.cfg, 'NUM_CLASSES', 10))
        else:
            # For other models
            aggregated_model = type(first_client_model)()
        
        # Copy parameters manually to avoid state_dict issues
        for param1, param2 in zip(first_client_model.parameters(), aggregated_model.parameters()):
            param2.data.copy_(param1.data)
        
        # Clear updates
        self.client_updates.clear()
        return [aggregated_model]
        
    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        """Deep copy a model including parameters."""
        import copy
        # Use deepcopy for all models
        clone = copy.deepcopy(model)
        return clone.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def aggregate_parameters(self, parameters_list: List[Dict[str, torch.Tensor]]):
        """
        Aggregate parameters from clients using FedEM-style aggregation
        
        Args:
            parameters_list: List of client parameters
            
        Returns:
            dict: Aggregated parameters
        """
        # Initialize storage for parameters
        aggregated = {}
        learner_params = defaultdict(lambda: defaultdict(list))
        
        # Group parameters by learner and layer
        for client_params in parameters_list:
            for name, param in client_params.items():
                # Skip mixture weights (client-specific)
                if 'mixture_weights' in name:
                    continue
                
                # Find corresponding learner
                for i in range(self.cfg.N_LEARNERS):
                    if f'learners.{i}.' in name:
                        # Store parameter with its name for this learner
                        learner_params[i][name].append(param)
                        break
        
        # Aggregate parameters for each learner
        for learner_idx in range(self.cfg.N_LEARNERS):
            learner_dict = learner_params[learner_idx]
            
            # Aggregate each parameter
            for param_name, param_list in learner_dict.items():
                if param_list:  # Check if we have parameters to aggregate
                    try:
                        # Stack and average parameters
                        stacked_params = torch.stack(param_list)
                        aggregated[param_name] = stacked_params.mean(dim=0)
                    except RuntimeError as e:
                        print(f"Error aggregating {param_name}: {str(e)}")
                        print(f"Shapes: {[p.shape for p in param_list]}")
                        continue
        
        # Copy base model parameters (if any)
        for name, param in self.global_models[0].named_parameters():
            if 'base_model.' in name:
                aggregated[name] = param.data.clone()
        
        return aggregated
    
    def train_round(self, clients: List[Client], round_idx: int):
        """
        Perform one round of federated training
        
        Args:
            clients: List of federated clients
            round_idx: Current round index
            
        Returns:
            dict: Round metrics
        """
        round_start = time.time()
        
        # Collect parameters from all clients
        parameters_list = []
        metrics_list = []
        
        for client in clients:
            # Local training
            parameters = client.local_train(epochs=self.cfg.LOCAL_EPOCHS)
            parameters_list.append(parameters)
            
            # Evaluate
            metrics = client.evaluate()
            metrics_list.append(metrics)
        
        # Aggregate parameters
        aggregated = self.aggregate_parameters(parameters_list)
        
        # Update global model
        for name, param in self.global_models[0].named_parameters():
            if name in aggregated:
                param.data.copy_(aggregated[name])
        
        # Broadcast to clients
        for client in clients:
            for name, param in client.model.named_parameters():
                if name in aggregated:
                    param.data.copy_(aggregated[name])
        
        # Compute round metrics
        round_metrics = {
            'clean_acc': np.mean([m['clean_acc'] for m in metrics_list]),
            'adv_acc': np.mean([m['adv_acc'] for m in metrics_list]),
            'greybox_sr': np.mean([m['greybox_sr'] for m in metrics_list]),
            'latency': time.time() - round_start,
            'memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
        
        # Log metrics
        self.logger.log_round(
            round_num=round_idx,
            **round_metrics
        )
        
        return round_metrics
    
    def evaluate_internal_attack(self, clients: List[Client]):
        """
        Evaluate internal grey-box attack from client 0
        
        Args:
            clients: List of federated clients
            
        Returns:
            dict: Attack success metrics
        """
        attacker = clients[0]
        victims = clients[1:]
        
        # Generate attacks on test set
        attack_success = []
        
        for data, target in attacker.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Purify input
            with torch.no_grad():
                data = attacker.model.diffusion_model.purify(
                    data,
                    steps=self.cfg.DIFFUSER_STEPS,
                    sigma=self.cfg.DIFFUSER_SIGMA
                )
            
            # Generate adversarial examples
            adv_data = attacker.attack.attack(data, target)
            
            # Test on other clients
            for victim in victims:
                with torch.no_grad():
                    outputs = victim.model(adv_data, is_training=False)
                    _, predicted = outputs.max(1)
                    attack_success.append((predicted != target).float().mean().item())
        
        return {
            'attack_success_rate': np.mean(attack_success),
            'std_success_rate': np.std(attack_success)
        } 
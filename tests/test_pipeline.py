import torch
import pytest
from pathlib import Path
from config import get_config
from federated.trainer import run_federated, is_diffuser_trained
from federated.client import FedClient
from defense.combined_defense import CombinedClassifier

def test_pipeline():
    """Smoke test for the complete defense pipeline."""
    # Use debug configuration
    cfg = get_config('debug')
    cfg.N_CLIENTS = 2
    cfg.BATCH_SIZE = 4
    
    # Clear any existing diffuser model
    diffuser_path = Path(cfg.model_dir) / 'diffusion' / 'diffuser.pth'
    if diffuser_path.exists():
        diffuser_path.unlink()
    
    # Run one round of training
    run_federated(cfg)
    
    # Verify DiffPure was trained
    assert is_diffuser_trained(cfg), "DiffPure model should be trained"
    
    # Load trained clients
    clients = [FedClient(i, cfg.dataset, cfg) for i in range(cfg.N_CLIENTS)]
    
    # Test purification effect with correct sigma
    test_data = torch.randn(4, 3, 32, 32).to(cfg.DEVICE)
    for client in clients:
        purified = client.classifier.purify_images(test_data, sigma=cfg.DIFFUSER_SIGMA)
        pixel_std = (purified - test_data).std()
        assert pixel_std > 0.02, "Purification should change pixel values significantly"
        
        # Test purification latency
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        _ = client.classifier.purify_images(test_data)
        end_time.record()
        
        torch.cuda.synchronize()
        latency = start_time.elapsed_time(end_time)
        assert latency < 200, f"Purification latency {latency:.2f}ms exceeds 200ms target"
    
    # Test clean accuracy
    clean_acc = test_clean_accuracy(clients, cfg)
    assert clean_acc > 0.7, f"Clean accuracy {clean_acc:.2f} should be above 70%"
    
    # Test grey-box attack success
    vanilla_success = test_grey_box_vanilla(clients, cfg)
    assert vanilla_success > 0.7, "Grey-box attack should succeed on vanilla FedAvg"
    
    defense_success = test_grey_box_defense(clients, cfg)
    assert defense_success < 0.4, "Combined defense should reduce grey-box success rate"
    
def test_clean_accuracy(clients, cfg):
    """Test clean accuracy on test set."""
    total_correct = 0
    total_samples = 0
    
    for client in clients:
        test_loader = torch.utils.data.DataLoader(
            client.test_data,
            batch_size=cfg.BATCH_SIZE
        )
        
        for data, target in test_loader:
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            with torch.no_grad():
                output = client.classifier(data)
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
    
    return total_correct / total_samples
    
def test_grey_box_vanilla(clients, cfg):
    """Test grey-box attack success on vanilla FedAvg."""
    # Disable defenses
    for client in clients:
        client.classifier.disable_defense = True
        
    # Run grey-box attack
    from attacks.internal_pgd import internal_attack
    return internal_attack(clients, cfg)
    
def test_grey_box_defense(clients, cfg):
    """Test grey-box attack success with combined defense."""
    # Enable defenses
    for client in clients:
        client.classifier.disable_defense = False
        
    # Run grey-box attack
    from attacks.internal_pgd import internal_attack
    return internal_attack(clients, cfg) 
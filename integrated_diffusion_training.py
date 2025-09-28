import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusion import UNet, Diffusion
from utils.data_utils import get_dataset
from pathlib import Path
import logging
from tqdm import tqdm
import os

def integrated_train_diffusion(cfg):
    """Train diffusion model directly without subprocess - FIXED ARCHITECTURE"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 INTEGRATED DIFFUSION TRAINING FOR CIFAR10 - FIXED ARCHITECTURE")
    
    device = torch.device(cfg.DEVICE)
    logger.info(f"Device: {device}")

    # Dataset-specific training optimizations
    if str(cfg.DATASET).lower() == 'cifar10':
        epochs = getattr(cfg, 'DIFFUSION_EPOCHS', 100) # Default to 100 for integrated
        batch_size = getattr(cfg, 'DIFFUSION_BATCH_SIZE', 32)
        lr = getattr(cfg, 'DIFFUSION_LR', 1e-4)
        logger.info(f"CIFAR-10 params: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    else:
        epochs = getattr(cfg, 'DIFFUSION_EPOCHS', 50)
        batch_size = getattr(cfg, 'DIFFUSION_BATCH_SIZE', 16)
        lr = getattr(cfg, 'DIFFUSION_LR', 1e-4)

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset, _ = get_dataset(cfg.DATASET, transform, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2)
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Define checkpoint path
    checkpoint_path = Path(f"checkpoints/diffuser_{cfg.DATASET.lower()}.pt")
    
    # Use EXACT same UNet architecture as in diffusion/diffuser.py - CRITICAL FIX!
    logger.info("🔧 Using correct UNet architecture from diffusion/diffuser.py")
    model = UNet(
        in_channels=cfg.IMG_CHANNELS, 
        hidden_channels=getattr(cfg, 'DIFFUSION_HIDDEN_CHANNELS', 128),
        use_additional_layers=False  # Keep it simple for CIFAR-10
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    diffusion = Diffusion(model, timesteps=1000, device=device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Log model architecture for verification
    logger.info("✅ Model architecture keys (should match main.py expectations):")
    sample_dict = model.state_dict()
    key_samples = list(sample_dict.keys())[:10]  # First 10 keys
    logger.info(f"Sample keys: {key_samples}")

    # Training loop
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Forward pass
            loss = diffusion.p_losses(images, None, loss_type="l1") # Simplified loss type
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(Loss=loss.item(), LR=optimizer.param_groups[0]['lr'])
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            temp_checkpoint = Path(f"checkpoints/diffuser_{cfg.DATASET.lower()}_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), temp_checkpoint)
            logger.info(f"Checkpoint saved: {temp_checkpoint}")

    # Save final model
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"✅ Diffusion training completed: {checkpoint_path}")
    
    # Verify saved model can be loaded
    try:
        test_model = UNet(
            in_channels=cfg.IMG_CHANNELS, 
            hidden_channels=getattr(cfg, 'DIFFUSION_HIDDEN_CHANNELS', 128),
            use_additional_layers=False
        )
        test_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        logger.info("✅ Model verification: Saved model can be loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Test standalone
    class TestConfig:
        DATASET = "cifar10"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        IMG_CHANNELS = 3
        DIFFUSION_EPOCHS = 5  # Short test
        DIFFUSION_BATCH_SIZE = 16
        DIFFUSION_LR = 1e-4
        DIFFUSION_HIDDEN_CHANNELS = 128
    
    cfg = TestConfig()
    integrated_train_diffusion(cfg)
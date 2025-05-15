# Technical Report: Combined Defense Implementation (DiffPure + pFedDef)

## 1. System Architecture

### 1.1 Core Components

The system implements a two-stage defense mechanism combining DiffPure and pFedDef:

1. **DiffPure Stage**:
   - Input: Raw or adversarial images
   - Process: Diffusion-based purification
   - Output: Purified images
   - Implementation: `defense/combined_defense.py`

2. **pFedDef Stage**:
   - Input: Purified images from DiffPure
   - Process: Personalized federated defense
   - Output: Final predictions
   - Implementation: `utils/pfeddef_utils.py`

### 1.2 Model Management

The system uses a centralized model management system (`utils/model_manager.py`) that handles:
- Model loading and saving
- Directory structure maintenance
- Model versioning and archiving
- Client-specific model paths

## 2. Implementation Details

### 2.1 DiffPure Implementation

```python
class CombinedDefense:
    def purify_images(self, images: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        # Add controlled noise
        noise = torch.randn_like(images) * sigma
        noisy_images = images + noise
        
        # Generate timesteps
        batch_size = images.shape[0]
        t = torch.ones(batch_size, device=self.device)
        
        # Predict and remove noise
        pred_noise = self.diffusion_model(noisy_images, t)
        purified_images = noisy_images - pred_noise
        
        return purified_images
```

Key Features:
- Noise level control via `sigma` parameter
- Batch processing support
- Device-agnostic implementation
- Memory-efficient with `torch.no_grad()`

### 2.2 pFedDef Implementation

```python
def train_pfeddef_model(model, train_loader, test_loader, attack, 
                       global_epochs=1, local_epochs=1, 
                       learning_rate=0.001, device='cuda'):
    # Optimizer configuration
    optimizer = torch.optim.SGD([
        {'params': model.learners[0].parameters(), 'lr': learning_rate},
        {'params': model.attention[0].parameters(), 'lr': learning_rate * 0.1},
        {'params': model.mixture_weights, 'lr': learning_rate * 0.01},
        {'params': model.normalize[0].parameters(), 'lr': learning_rate * 0.1},
        {'params': model.feature_extractors[0].parameters(), 'lr': learning_rate * 0.1}
    ], momentum=0.9, weight_decay=CONFIG['model']['pfeddef']['weight_decay'])
```

Key Features:
- Multi-component optimization
- Attention mechanism
- Feature extraction
- Mixture weights for ensemble

### 2.3 Combined Defense Pipeline

```python
def defend(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Step 1: Purify images using DiffPure
    purified_images = self.purify_images(images)
    
    # Step 2: Apply pFedDef defense
    self.pfeddef_model.eval()
    with torch.no_grad():
        predictions = self.pfeddef_model(purified_images)
    
    return purified_images, predictions
```

## 3. Training Process

### 3.1 DiffPure Training

1. **Model Architecture**:
   - UNet-based diffusion model
   - Input channels: 3 (RGB)
   - Hidden channels: 64
   - Output: Noise prediction

2. **Training Parameters**:
   - Sigma: 0.1 (default)
   - Batch size: Configurable
   - Learning rate: 1e-4
   - Optimizer: Adam

### 3.2 pFedDef Training

1. **Model Architecture**:
   - Base: ResNet18
   - Modifications:
     - Custom attention layers
     - Feature extractors
     - Normalization layers
     - Ensemble mechanism

2. **Training Parameters**:
   - Global epochs: 1
   - Local epochs: 1
   - Learning rate: 0.001
   - Momentum: 0.9
   - Weight decay: Configurable

## 4. Evaluation System

### 4.1 Attack Implementation

```python
attacks = [
    PGDAttack(
        epsilon=8/255,
        alpha=2/255,
        steps=10,
        random_start=True
    ),
    FGSMAttack(
        epsilon=8/255
    )
]
```

### 4.2 Metrics Collection

```python
results = {
    'clean_acc': 0.0,
    'pgd_acc': 0.0,
    'fgsm_acc': 0.0
}
```

## 5. Current Limitations

1. **Training Efficiency**:
   - Limited to 1-2 epochs for quick testing
   - No distributed training support yet

2. **Model Architecture**:
   - Fixed ResNet18 backbone
   - Limited model customization options

3. **Attack Coverage**:
   - Currently supports only PGD and FGSM
   - No adaptive attack testing

## 6. Future Improvements

1. **Architecture Enhancements**:
   - Support for multiple model architectures
   - Dynamic model selection
   - Improved ensemble mechanisms

2. **Training Optimizations**:
   - Distributed training support
   - Advanced learning rate scheduling
   - Better early stopping criteria

3. **Defense Mechanisms**:
   - Additional attack types
   - Adaptive defense strategies
   - Improved purification methods

## 7. Code Structure

```
.
├── defense/
│   └── combined_defense.py    # Main defense implementation
├── models/
│   ├── target/               # Target model files
│   ├── diffusion/            # Diffusion model files
│   └── pfeddef/             # pFedDef model files
├── utils/
│   ├── model_manager.py      # Model management
│   ├── data_utils.py         # Data handling
│   └── pfeddef_utils.py      # pFedDef utilities
├── attacks/
│   ├── pgd.py               # PGD attack
│   └── fgsm.py              # FGSM attack
└── config.py                # Configuration
```

## 8. Configuration System

The system uses a centralized configuration system (`config.py`) that manages:
- Model parameters
- Training settings
- Attack configurations
- Defense parameters
- Dataset settings

## 9. Testing and Validation

1. **Unit Tests**:
   - Model loading/saving
   - Defense pipeline
   - Attack implementations

2. **Integration Tests**:
   - End-to-end defense
   - Attack-defense scenarios
   - Performance metrics

## 10. Performance Considerations

1. **Memory Usage**:
   - Batch size management
   - Model checkpointing
   - Gradient accumulation

2. **Computation Time**:
   - GPU utilization
   - Batch processing
   - Model optimization

## 11. Documentation

The project includes:
- Code documentation
- API references
- Usage examples
- Configuration guides

## 12. Next Steps

1. **Immediate Tasks**:
   - Complete attack implementations
   - Enhance model architectures
   - Improve training process

2. **Future Development**:
   - Distributed training
   - Advanced defense mechanisms
   - Extended attack coverage 
# Model Organization

This directory contains all model checkpoints organized in a standardized structure.

## Directory Structure

```
models/
├── diffusion/           # Diffusion model checkpoints
├── target/             # Target model checkpoints
├── pfeddef/            # PFedDef model checkpoints
│   ├── target_models/  # Target models used in PFedDef
│   └── transfer_models/# Transfer learning models
└── client_weights/     # Client-specific model weights
```

## Naming Conventions

### Diffusion Models
Format: `{dataset}_{model_type}_defense_diffusion.pth`
Example: `cifar10_ddpm_defense_diffusion.pth`

For client-specific diffusion models:
Format: `{dataset}_{model_type}_client_{id}_sigma{value}_steps{count}.pth`
Example: `cifar10_ddpm_client_0_sigma0.1_steps100.pth`

### Client Models
Format: `{dataset}_{architecture}_client_{id}.pth`
Example: `cifar10_resnet18_client_0.pth`

### PFedDef Models
Format: `{dataset}_{model_type}_defense_pfeddef.pth`
Example: `cifar10_resnet18_defense_pfeddef.pth`

## Model Types
- `ddpm`: Denoising Diffusion Probabilistic Model
- `resnet18`: ResNet-18 Architecture
- `vgg16`: VGG-16 Architecture

## Datasets
- `cifar10`: CIFAR-10 Dataset
- `mnist`: MNIST Dataset

## Parameters
- `sigma`: Noise level for diffusion models
- `steps`: Number of diffusion steps
- `client_id`: Identifier for client-specific models (0-9)

## Notes
- Backup files are stored with `.bak` extension
- Each model type has its dedicated directory for better organization
- Client weights are consolidated under the `client_weights` directory
- Model configurations should be verified before training 
# Project Dependency Map

## Core Files

### `main.py`
- **Functions**: 
  - `setup_logging` - Configure logging
  - `save_metrics` - Save metrics to CSV
  - `plot_metric_curve` - Plot metric curves
  - `save_purifier_vis` - Save visualization of purified images
  - `evaluate_defense` - Evaluate defense against adversarial attacks
  - `run_federated` - Run federated training with combined defense
  - `main` - Main entry point
- **Imports**:
  - `config` (`get_config`, `parse_args`, `DEVICE`)
  - `diffusion.diffuser.UNet`
  - `federated.client.Client`
  - `federated.server.Server`
  - `attacks.pgd.PGDAttack`
  - `attacks.internal_pgd.InternalPGDAttack`
  - `models.pfeddef_model.pFedDefModel`

### `config.py`
- **Classes**: 
  - `Config` - Configuration dataclass
- **Functions**:
  - `get_config` - Get configuration based on preset
  - `parse_args` - Parse command-line arguments
- **Constants**:
  - `BASE_DIR`, `DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `LOGS_DIR`, `CHECKPOINTS_DIR`
  - `USE_CUDA`, `DEVICE`
  - `PRESETS` - Dictionary of preset configurations

### `quick_test.py`
- **Functions**: 
  - `optimize_memory` - Optimize PyTorch memory usage
  - `main` - Main testing function
- **Imports**:
  - `config` (`get_config`, `DEVICE`)
  - `diffusion.diffuser.UNet`
  - `models.pfeddef_model.pFedDefModel`
  - `attacks.pgd.PGDAttack`
  - `defense.combined_defense.CombinedClassifier`

## Modules

### Defense Module
- **`defense/combined_defense.py`**
  - **Classes**: 
    - `CombinedDefense` - Combined DiffPure and pFedDef defense
    - `CombinedClassifier` - NN module implementing the combined defense
  - **Imports**:
    - `utils.model_manager.ModelManager`
    - `diffusion.diffuser.UNet`
    - `models.resnet.ResNet18`

### Diffusion Module
- **`diffusion/diffuser.py`**
  - **Classes**: 
    - `DoubleConv` - Double convolution block
    - `Down` - Downsampling block
    - `Up` - Upsampling block
    - `UNet` - UNet model for diffusion
    - `DiffusionPurifier` - DPM-Solver-2 based diffusion purifier
  - **Imports**: None (only PyTorch)

### Federated Module
- **`federated/client.py`**
  - **Imports**: Need to be analyzed
- **`federated/server.py`**
  - **Imports**: Need to be analyzed
- **`federated/trainer.py`**
  - **Imports**: Need to be analyzed

### Attacks Module
- **`attacks/pgd.py`**
  - **Classes**: 
    - `PGDAttack` - PGD Attack implementation
  - **Imports**: None (only PyTorch)
- **`attacks/internal_pgd.py`**
  - **Imports**: Need to be analyzed
- **`attacks/fgsm.py`**
  - **Imports**: Need to be analyzed

### Models Module
- **`models/pfeddef_model.py`**
  - **Classes**: 
    - `pFedDefModel` - pFedDef model implementation with ensemble of learners
  - **Imports**: None (only PyTorch)
- **`models/resnet.py`**
  - **Classes**: 
    - `ResNet18` - ResNet18 model
  - **Imports**: Need to be analyzed

### Metrics Module
- **`metrics/logger.py`**
  - **Imports**: Need to be analyzed

### Scripts
- **`scripts/sanity_suite.py`**
  - **Classes**: 
    - `QuickDebugConfig` - Debug configuration with reduced settings
  - **Functions**: 
    - `optimize_memory` - Optimize memory usage
    - `compute_pixel_std_diff` - Compute standard deviation of pixel differences
    - `compute_learner_cos_sim` - Compute cosine similarity between learners
    - `compute_param_drift` - Compute parameter drift between models
    - `create_vanilla_fedavg` - Create vanilla FedAvg model
    - `plot_learner_logits_hist` - Plot learner logits histograms
    - `plot_grey_sr_curve` - Plot grey-box success rate curve
    - `save_purifier_vis` - Save purifier visualization
    - `main` - Main function
  - **Imports**:
    - `config` (`Config`, `DEVICE`)
    - `models.pfeddef_model.pFedDefModel`
    - `diffusion.diffuser.UNet`
    - `attacks.pgd.PGDAttack`
    - `federated.client.Client`
    - `federated.server.Server`

## Dependency Graph

```
main.py
├── config.py
├── diffusion/
│   └── diffuser.py
├── federated/
│   ├── client.py
│   └── server.py
├── attacks/
│   ├── pgd.py
│   └── internal_pgd.py
└── models/
    └── pfeddef_model.py

quick_test.py
├── config.py
├── diffusion/diffuser.py
├── models/pfeddef_model.py
├── attacks/pgd.py
└── defense/combined_defense.py

defense/combined_defense.py
├── utils/model_manager.py
├── diffusion/diffuser.py
└── models/resnet.py

scripts/sanity_suite.py
├── config.py
├── models/pfeddef_model.py
├── diffusion/diffuser.py
├── attacks/pgd.py
├── federated/client.py
└── federated/server.py
``` 
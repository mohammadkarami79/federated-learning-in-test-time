# Combined Defense: DiffPure + pFedDef

This project implements a robust combined defense mechanism for federated learning that integrates DiffPure (diffusion-based purification) with pFedDef (personalized federated defense). This hybrid approach provides strong protection against adversarial attacks while maintaining high performance on clean data.

## Project Overview

The combined defense works in two stages:
1. **DiffPure**: Uses diffusion models to purify input images by removing adversarial perturbations
2. **pFedDef**: Employs personalized, multi-learner federated training to create robust, attack-resistant models
3. **MAE Gatekeeper**: Optional Masked Autoencoder detector for adversarial example identification

This approach is particularly effective against:
- PGD (Projected Gradient Descent) attacks
- FGSM (Fast Gradient Sign Method) attacks
- Transfer attacks from unknown models

## Project Structure

```
.
├── attacks/              # Adversarial attack implementations
│   ├── pgd.py            # PGD attack implementation
│   ├── fgsm.py           # FGSM attack implementation 
│   └── internal_pgd.py   # Internal PGD attack variant
├── defense/              # Defense mechanisms
│   ├── combined_defense.py  # Combined DiffPure + pFedDef implementation
│   └── mae_detector.py   # MAE-based adversarial example detector
├── diffusion/            # Diffusion model components
│   └── diffuser.py       # UNet-based diffusion model for purification
├── extensions/           # Extension hooks for custom defense components
│   └── sample_idea/      # Example extension template
├── federated/            # Federated learning components
│   ├── client.py         # Client implementation
│   ├── server.py         # Server aggregation logic
│   └── trainer.py        # Federated training orchestration
├── models/               # Model definitions and saved weights
│   ├── pfeddef_model.py  # pFedDef model implementation
│   └── resnet.py         # ResNet backbone models
├── scripts/              # Utility scripts
│   ├── sanity_suite.py   # Quick sanity check for system integrity
│   └── train_mae_detector.py  # Standalone script to train MAE detector
├── utils/                # Utility functions
│   ├── data_utils.py     # Dataset loading and processing
│   └── model_manager.py  # Model loading/saving utilities
├── checkpoints/          # Model checkpoints (generated during training)
├── results/              # Results and visualizations (generated during runs)
├── logs/                 # Log files (generated during runs)
├── config.py             # Configuration settings
├── main.py               # Main execution script
├── train_diffpure.py     # Standalone script to train diffusion model
└── requirements.txt      # Python dependencies
```

## Setup and Requirements

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 1.9+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pFedDef_v1_kaggle.git
   cd pFedDef_v1_kaggle
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories (if not present):
   ```bash
   mkdir -p checkpoints results logs
   ```

## Usage

### 1. Train the DiffPure UNet Purifier (First Time or New Dataset)

```bash
python train_diffpure.py
```
This trains the diffusion model used for input purification and saves it to `checkpoints/diffuser.pt`.

**Important**: This step is only needed once per dataset or if you want to optimize the purifier. The `main.py` script will load this model but not train it.

### 2. Train the MAE Detector (Optional)

```bash
python scripts/train_mae_detector.py --preset debug   # For quick testing
python scripts/train_mae_detector.py --preset full    # For full training
```

This trains the Masked Autoencoder detector for identifying adversarial examples, which enables the system to selectively apply defensive measures. The detector is saved to `checkpoints/mae_detector.pth`.

### 3. Run the Combined Defense Pipeline

#### For Quick Testing/Debugging:
```bash
python main.py --preset debug
```

This will:
- Load the DiffPure model from `checkpoints/diffuser.pt`
- Load the MAE detector (if enabled in config)
- Create and train federated clients with pFedDef models (2 clients, 2 rounds)
- Perform model aggregation after each round
- Evaluate against adversarial attacks
- Generate metrics and visualizations in the `results/` directory

#### For Full Experiments:
```bash
python main.py --preset full
```

Runs the complete training pipeline with more clients (5), more learners per client (3), and more rounds (100) for a comprehensive evaluation. This may take several hours to complete.

### 4. Quick Sanity Test

To run a lightweight test of the system:
```bash
python scripts/sanity_suite.py --preset debug
```
This performs basic checks on all system components and verifies they're working correctly.

## Defense Mechanisms Explained

### DiffPure (Diffusion Purification)
1. **Working Principle**: Adds controlled noise to potentially adversarial inputs, then uses a diffusion model to predict and remove both the added noise and adversarial perturbations
2. **Advantages**: Effective against various attack types without needing to know attack specifics
3. **Implementation**: UNet architecture in `diffusion/diffuser.py` trained via `train_diffpure.py`

### pFedDef (Personalized Federated Defense)
1. **Working Principle**: Trains multiple personalized models per client with ensemble diversity
2. **Key Components**:
   - Multiple learners per client that see different data views
   - KL-divergence regularization to maintain learner diversity
   - Federated aggregation of learner pools across clients
3. **Implementation**: Core logic in `models/pfeddef_model.py` and `federated/` directory

### MAE Detector (Adversarial Gatekeeper)
1. **Working Principle**: Uses a Masked Autoencoder to compute reconstruction error of inputs
2. **Key Benefits**:
   - Only applies expensive defenses when necessary
   - Enables selective application of defensive measures
   - Improves system efficiency without sacrificing security
3. **Implementation**: Detector model in `defense/mae_detector.py` trained via `scripts/train_mae_detector.py`

### Combined Pipeline
The defense pipeline integrates all approaches:
1. MAE detector identifies potential adversarial examples (if enabled)
2. Input images are purified through the diffusion model (DiffPure)
3. Purified images are processed by the ensemble of pFedDef learners
4. Ensemble predictions combine for the final classification

## Configuration Options

All configuration parameters are defined in `config.py` with two preset configurations:

### Key Configuration Parameters

| Parameter | Description | Default (debug) | Default (full) |
|-----------|-------------|----------------|----------------|
| N_CLIENTS | Number of federated clients | 2 | 5 |
| N_LEARNERS | Number of learners per client | 2 | 3 |
| N_ROUNDS | Number of federated rounds | 2 | 100 |
| BATCH_SIZE | Batch size for training | 32 | 64 |
| DIFFUSER_STEPS | Steps in diffusion process | 4 | 10 |
| DIFFUSER_SIGMA | Diffusion noise level | 0.04 | 0.1 |
| DATASET_NAME | Dataset to use | "CIFAR10" | "CIFAR10" |
| DATA_PATH | Path to dataset files | "./data" | "./data" |
| USE_AMP | Use automatic mixed precision | True | False |
| RESNET_WIDTH | Width multiplier for ResNet models | 0.5 | 1.0 |
| ENABLE_MAE_DETECTOR | Enable MAE adversarial detector | True | True |
| MAE_THRESHOLD | MSE threshold for adversarial detection | 0.1 | 0.1 |
| EPOCHS_DETECTOR | Epochs for training MAE detector | 1 | 5 |

To override any setting, you can customize `config.py` or create a new preset.

## Dataset Configuration

The system is configured to work with CIFAR10 by default, but can be extended to other datasets:

```python
# In config.py presets:
'DATASET_NAME': 'CIFAR10',   # Options: "CIFAR10", "MedMNIST", etc.
'DATA_PATH': './data'        # Root directory for datasets
```

To use a different dataset:
1. Update these parameters in `config.py`
2. Ensure the dataset is supported in `utils/data_utils.py`
3. Retrain the diffusion model: `python train_diffpure.py`
4. Retrain the MAE detector: `python scripts/train_mae_detector.py --preset debug`
5. Run the main pipeline: `python main.py --preset debug`

## Expected Results

When running with the default configurations, you should expect:

### Debug Preset
- Clean accuracy: ~70-85%
- Grey-box attack success rate: ≤ 0.40 (lower is better)
- Runtime: ~40 minutes on an RTX 3060

### Full Preset
- Clean accuracy: ~85-95%
- Grey-box attack success rate: ≤ 0.30 (lower is better)
- Runtime: Several hours (overnight run recommended)

## Output Files

After running the pipeline, check these locations:
- `results/metrics.csv` - Round-by-round performance metrics
- `results/grey_sr_curve.png` - Attack success rate over time
- `results/clean_acc_curve.png` - Clean accuracy over time
- `results/purifier_vis.png` - Comparison of original vs. purified images
- `logs/train.log` - Detailed training logs

## Extension System

To implement custom defense mechanisms, use the extension system:

1. Create a new directory under `extensions/` with your extension name
2. Implement a custom defense hook by inheriting from `CombinedClassifier`
3. Set `ENABLE_<YOUR_EXTENSION>=True` in the config presets
4. See `extensions/sample_idea/` for a template

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in config
   - Enable `USE_AMP=True` for mixed precision
   - Reduce `DIFFUSER_STEPS` to minimize memory usage
   - Decrease `RESNET_WIDTH` to use smaller models

2. **Gradient Computation Errors**
   - Check input tensors have `requires_grad=True` set properly
   - Verify model is in training mode when appropriate
   - Look for attempts to compute gradients within `torch.no_grad()` contexts

3. **Poor Defense Performance**
   - Ensure `checkpoints/diffuser.pt` exists and was trained on the correct dataset
   - Try increasing `DIFFUSER_SIGMA` for stronger purification
   - Increase `N_LEARNERS` for better ensemble diversity

## Performance Optimization

For faster training/evaluation:
1. Use the `debug` preset for quick iterations
2. Enable mixed precision with `USE_AMP=True`
3. Reduce model size with `RESNET_WIDTH=0.5` or lower
4. Use fewer diffusion steps with `DIFFUSER_STEPS=4`

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DiffPure: [Original Paper](https://arxiv.org/abs/2203.09871)
- pFedDef: [Original Paper](https://arxiv.org/abs/2103.03228)
- CIFAR-10 dataset: [Website](https://www.cs.toronto.edu/~kriz/cifar.html) 
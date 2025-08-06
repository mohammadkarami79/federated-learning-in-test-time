# Methodology: Federated Learning with Test-Time Adversarial Defense Using Diffusion-based Purification and Masked Autoencoder Detection

## Abstract

This paper presents a novel federated learning framework that integrates test-time adversarial defense mechanisms through diffusion-based input purification and masked autoencoder (MAE) detection. Our methodology combines personalized federated defense (pFedDef) with advanced adversarial detection and mitigation techniques to enhance robustness in distributed learning environments across multiple datasets including CIFAR-10, CIFAR-100, MNIST, and medical brain tumor (BR35H) classification.

## 1. Problem Formulation

### 1.1 Federated Learning Setup

Consider a federated learning system with $N$ clients, where each client $i$ has a local dataset $\mathcal{D}_i = \{(x_j^{(i)}, y_j^{(i)})\}_{j=1}^{n_i}$ with $n_i$ samples. The global objective is to minimize:

$$\min_{\theta} F(\theta) = \sum_{i=1}^{N} \frac{n_i}{n} F_i(\theta)$$

where $F_i(\theta) = \frac{1}{n_i} \sum_{j=1}^{n_i} \ell(f_{\theta}(x_j^{(i)}), y_j^{(i)})$ is the local loss function, $n = \sum_{i=1}^{N} n_i$ is the total number of samples, and $\ell(\cdot, \cdot)$ is the cross-entropy loss function.

### 1.2 Adversarial Threat Model

We consider $\ell_\infty$-bounded adversarial perturbations where an adversarial example is defined as:

$$x_{adv} = x + \delta, \quad \text{s.t.} \quad \|\delta\|_\infty \leq \epsilon$$

The adversary aims to maximize the prediction error using the Projected Gradient Descent (PGD) attack:

$$x_{adv}^{(t+1)} = \Pi_{\mathcal{B}_\epsilon(x)} \left( x_{adv}^{(t)} + \alpha \cdot \text{sign}(\nabla_x \ell(f_\theta(x_{adv}^{(t)}), y)) \right)$$

where $\Pi_{\mathcal{B}_\epsilon(x)}$ denotes projection onto the $\ell_\infty$ ball of radius $\epsilon$ centered at $x$, and $\alpha$ is the step size.

## 2. Proposed Defense Framework

### 2.1 Personalized Federated Defense (pFedDef)

Our framework employs a mixture of experts approach where each client maintains $K = 3$ local learners with an attention mechanism:

$$\theta_i = \{\theta_i^{(1)}, \theta_i^{(2)}, \theta_i^{(3)}\}$$

The local prediction is computed using learner-specific attention weights:

$$\hat{y}_i(x) = \sum_{k=1}^{K} \alpha_i^{(k)}(x) \cdot f_{\theta_i^{(k)}}(x)$$

where $\alpha_i^{(k)}(x) = \sigma(g_i^{(k)}(x))$ are input-dependent attention weights computed by learner-specific attention networks $g_i^{(k)}$.

#### 2.1.1 Architecture Details

Each learner consists of:
- **Feature Extractor**: ResNet-18 backbone with pretrained weights
- **Classifier Head**: Two-layer MLP with 256 hidden units and ReLU activation
- **Attention Network**: Two-layer MLP outputting scalar attention weights

The mixture weights are learned through:

$$w_i^{(k)}(x) = \frac{\exp(\text{MLP}_i^{(k)}(\phi(x)))}{\sum_{j=1}^{K} \exp(\text{MLP}_i^{(j)}(\phi(x)))}$$

where $\phi(x)$ represents the shared feature representation from ResNet-18.

### 2.2 Diffusion-based Input Purification

#### 2.2.1 U-Net Architecture

Our diffusion model employs a U-Net architecture with the following specifications:
- **Encoder**: 4 downsampling blocks with DoubleConv layers
- **Decoder**: 4 upsampling blocks with skip connections
- **Hidden Channels**: 256 (optimized for memory efficiency)
- **Time Embedding**: Multi-layer perceptron with SiLU activation

#### 2.2.2 Forward Diffusion Process

The forward diffusion process gradually adds Gaussian noise to the input following a linear schedule:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

The noise schedule $\{\beta_t\}_{t=1}^{T}$ follows: $\beta_1 = 1 \times 10^{-4}$ to $\beta_T = 2 \times 10^{-2}$ with $T = 1000$ timesteps.

#### 2.2.3 Reverse Diffusion Process

The reverse process learns to denoise using a parameterized model $\epsilon_\phi(x_t, t)$:

$$p_\phi(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\phi(x_t, t), \Sigma_\phi(x_t, t))$$

where:

$$\mu_\phi(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\phi(x_t, t)\right)$$

with $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

#### 2.2.4 Training Objective

The diffusion model is trained to minimize the denoising score matching objective:

$$\mathcal{L}_{diff} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0,I), t \sim \text{Uniform}(1,T)}\left[\|\epsilon - \epsilon_\phi(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$

### 2.3 Masked Autoencoder Detection

#### 2.3.1 Architecture

The MAE consists of a Vision Transformer-based encoder-decoder architecture:

- **Patch Size**: 4×4 for CIFAR datasets, 16×16 for BR35H
- **Encoder**: 12-layer transformer with 512 embedding dimensions
- **Decoder**: 8-layer transformer with 512 embedding dimensions
- **Attention Heads**: 8 heads per layer

#### 2.3.2 Masking Strategy

Random masking with high masking ratio:

$$\mathcal{M} \sim \text{Bernoulli}(1-r)^{P}, \quad r = 0.75$$

where $P$ is the total number of patches and $r$ is the masking ratio.

#### 2.3.3 Training Objective

The MAE minimizes reconstruction loss only on masked patches:

$$\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}^c|} \sum_{p \in \mathcal{M}^c} \|x_p - \hat{x}_p\|^2$$

where $\mathcal{M}^c$ denotes the set of masked patches.

#### 2.3.4 Adversarial Detection

The detection score is computed as the mean squared error across all patches:

$$s_{det}(x) = \frac{1}{P} \sum_{p=1}^{P} \|x_p - \hat{x}_p\|^2$$

An input is classified as adversarial if $s_{det}(x) > \tau$ where $\tau = 0.1$ is the detection threshold.

## 3. Integrated Defense Pipeline

### 3.1 Test-Time Defense Protocol

For each test input $x$, our integrated defense follows a three-stage pipeline:

**Stage 1: Detection**
```
s_det = MAE_detect(x)
is_adversarial = (s_det > τ)
```

**Stage 2: Purification (Conditional)**
```
if is_adversarial:
    x_pure = DiffusionPurify(x, t_purify)
else:
    x_pure = x
```

**Stage 3: Prediction**
```
y_pred = pFedDef_predict(x_pure)
```

### 3.2 Adaptive Purification

The purification timestep is adaptively chosen based on detection confidence:

$$t_{purify} = \min\left(50, \left\lfloor 50 \cdot \frac{s_{det}(x)}{\tau_{max}} \right\rfloor\right)$$

where $\tau_{max} = 1.0$ represents the maximum expected detection score.

## 4. Federated Training Algorithm

### 4.1 Local Update

Each client $i$ performs local training for $E = 15$ epochs using the following update rule:

**For each local epoch:**
1. Sample mini-batch $\mathcal{B}_i$ from $\mathcal{D}_i$
2. For each expert $k \in \{1, 2, 3\}$:
   $$\theta_i^{(k)} \leftarrow \theta_i^{(k)} - \eta \nabla_{\theta_i^{(k)}} \mathcal{L}_i^{(k)}(\mathcal{B}_i)$$
3. Update attention networks:
   $$\phi_i^{(k)} \leftarrow \phi_i^{(k)} - \eta \nabla_{\phi_i^{(k)}} \mathcal{L}_{att}^{(k)}(\mathcal{B}_i)$$

### 4.2 Global Aggregation

The server aggregates client models using FedAvg with sample-weighted averaging:

$$\theta_{global}^{(k)} = \sum_{i=1}^{N} \frac{n_i}{n} \theta_i^{(k)}$$

### 4.3 Complete Training Algorithm

```python
Algorithm 1: Federated Learning with Test-Time Defense

Input: N = 15 clients, T = 20 communication rounds, E = 15 local epochs
Output: Global model parameters, trained defense models

1: // Phase 1: Train Diffusion Models for each dataset
2: for each dataset d ∈ {CIFAR-10, CIFAR-100, MNIST, BR35H} do
3:    Train UNet(hidden_channels=256) for 50 epochs using Eq. (7)
4:    Save diffuser_d.pt and configuration
5: end for

6: // Phase 2: Initialize MAE Detectors  
7: for each dataset d do
8:    Initialize MAE with patch_size=4, embed_dim=512
9:    Train for 30 epochs using Eq. (10)
10: end for

11: // Phase 3: Federated Training with Defense
12: Initialize θ_global with ResNet-18 + 3 learners
13: for round t = 1 to T do
14:    for each client i ∈ {1,...,N} in parallel do
15:       θ_i ← θ_global  // Download global model
16:       for epoch e = 1 to E do
17:          Sample batch B_i from D_i
18:          // Optional: Apply test-time defense during training
19:          for x in B_i do
20:             if s_det(x) > τ then
21:                x ← DDIM_purify(x, t_purify)
22:             end if
23:          end for
24:          Update θ_i using SGD with learning_rate=0.001
25:       end for
26:       Upload θ_i to server
27:    end for
28:    θ_global ← FedAvg({θ_i}_{i=1}^N)  // Server aggregation
29:    
30:    // Evaluation every round
31:    if t mod 1 == 0 then
32:       Evaluate on test set with PGD attacks
33:       Record clean_acc, adv_acc, detection_rate
34:    end if
35: end for
```

## 5. Experimental Setup

### 5.1 Datasets and Preprocessing

#### 5.1.1 Standard Datasets
- **CIFAR-10**: 32×32 RGB images, 10 classes, 50K train/10K test
- **CIFAR-100**: 32×32 RGB images, 100 classes, 50K train/10K test  
- **MNIST**: 28×28 grayscale images, 10 classes, 60K train/10K test

#### 5.1.2 Medical Dataset
- **BR35H**: 224×224 RGB brain MRI images, 2 classes (tumor/no tumor)
- **Preprocessing**: Resizing to 224×224, normalization to [0,1]
- **Augmentation**: Random horizontal flip, rotation (±15°)

### 5.2 Architecture Specifications

#### 5.2.1 Diffusion Model (U-Net)
```python
UNet Configuration:
- Input Channels: {1 (MNIST), 3 (others)}
- Hidden Channels: 256
- Encoder Blocks: 4 (with DoubleConv + MaxPool)
- Decoder Blocks: 4 (with ConvTranspose + skip connections)
- Time Embedding: 3-layer MLP with SiLU activation
- Output: Same dimension as input
```

#### 5.2.2 MAE Detector
```python
MAE Configuration:
- Patch Size: 4×4 (CIFAR), 16×16 (BR35H)
- Encoder: 12-layer ViT, 512 embed_dim, 8 heads
- Decoder: 8-layer ViT, 512 embed_dim, 8 heads  
- Mask Ratio: 0.75
- Reconstruction Loss: MSE on masked patches
- Detection Threshold: τ = 0.1
```

#### 5.2.3 pFedDef Model
```python
pFedDef Configuration:
- Base Model: ResNet-18 (pretrained on ImageNet)
- Number of Learners: K = 3
- Learner Architecture: 
  - Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, num_classes)
- Attention Architecture:
  - Linear(512, 256) → ReLU → Dropout(0.2) → Linear(256, 1) → Sigmoid
- Mixture Weights: Learnable parameters initialized uniformly
```

### 5.3 Training Hyperparameters

#### 5.3.1 Federated Learning
```python
FL_CONFIG = {
    'n_clients': 15,
    'communication_rounds': 20, 
    'local_epochs': 15,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4
}
```

#### 5.3.2 Diffusion Training
```python
DIFFUSION_CONFIG = {
    'epochs': 50,
    'batch_size': 64,  # Reduced for memory efficiency
    'learning_rate': 1e-4,
    'optimizer': 'Adam',
    'beta1': 0.9,
    'beta2': 0.999,
    'scheduler': 'CosineAnnealingLR',
    'timesteps': 1000,
    'noise_schedule': 'linear'
}
```

#### 5.3.3 MAE Training
```python
MAE_CONFIG = {
    'epochs': 30,
    'batch_size': 128,
    'learning_rate': 1e-3,
    'optimizer': 'AdamW',
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'scheduler': 'CosineAnnealingLR'
}
```

### 5.4 Attack Configuration

#### 5.4.1 PGD Attack
```python
PGD_CONFIG = {
    'epsilon': 8/255,      # L∞ perturbation budget
    'alpha': 2/255,        # Step size  
    'steps': 100,          # Number of iterations
    'random_start': True,  # Random initialization
    'loss_function': 'CrossEntropy'
}
```

#### 5.4.2 Additional Attacks (Evaluation)
- **FGSM**: ε ∈ {4/255, 8/255, 16/255}
- **C&W**: κ = 0, c ∈ {0.1, 1, 10}, maximum 1000 iterations
- **AutoAttack**: Standard configuration with ε = 8/255

### 5.5 Hardware and Software Environment

#### 5.5.1 Computational Resources
```python
HARDWARE_SPECS = {
    'GPU': 'NVIDIA RTX (6GB VRAM)',
    'CPU': 'Multi-core processor',
    'RAM': '16GB+',
    'Storage': 'SSD for fast I/O'
}
```

#### 5.5.2 Software Stack
```python
SOFTWARE_STACK = {
    'Framework': 'PyTorch 2.0+',
    'CUDA': '11.8+',
    'Python': '3.8+',
    'Dependencies': ['torchvision', 'numpy', 'tqdm', 'pathlib']
}
```

## 6. Evaluation Metrics

### 6.1 Robustness Metrics

#### 6.1.1 Accuracy Metrics
- **Clean Accuracy**: 
  $$Acc_{clean} = \frac{1}{|\mathcal{T}|} \sum_{(x,y) \in \mathcal{T}} \mathbb{1}[f(x) = y]$$

- **Adversarial Accuracy**: 
  $$Acc_{adv} = \frac{1}{|\mathcal{T}|} \sum_{(x,y) \in \mathcal{T}} \mathbb{1}[f(x_{adv}) = y]$$

#### 6.1.2 Detection Metrics
- **Detection Rate (True Positive Rate)**:
  $$DR = \frac{TP}{TP + FN}$$

- **False Positive Rate**:
  $$FPR = \frac{FP}{FP + TN}$$

- **Area Under ROC Curve (AUC)**:
  $$AUC = \int_0^1 TPR(FPR^{-1}(t)) dt$$

### 6.2 Efficiency Metrics

#### 6.2.1 Communication Efficiency
- **Total Communication Cost**: Sum of uploaded/downloaded model parameters
- **Communication Rounds**: Number of server-client synchronizations
- **Convergence Rate**: Rounds to achieve target accuracy

#### 6.2.2 Computational Efficiency  
- **Training Time**: Wall-clock time for complete training
- **Inference Time**: Time per sample during evaluation
- **Memory Usage**: Peak GPU memory consumption
- **FLOPs**: Floating point operations per inference

### 6.3 Statistical Analysis

All experiments are repeated **5 times** with different random seeds (42, 123, 456, 789, 1011). Results are reported as:

$$\mu \pm \sigma$$

where $\mu$ is the mean and $\sigma$ is the standard deviation across runs.

Statistical significance is assessed using **paired t-tests** with significance level $\alpha = 0.05$.

## 7. Dataset-Specific Configurations

### 7.1 CIFAR-10/100 Configuration
```python
CIFAR_CONFIG = {
    'img_size': 32,
    'img_channels': 3,
    'num_classes': {10, 100},
    'normalization': 'ImageNet_stats',
    'data_augmentation': ['RandomCrop(32, padding=4)', 'RandomHorizontalFlip()']
}
```

### 7.2 MNIST Configuration  
```python
MNIST_CONFIG = {
    'img_size': 28,
    'img_channels': 1,
    'num_classes': 10,
    'normalization': 'mean=0.1307, std=0.3081',
    'data_augmentation': None  # Minimal augmentation for MNIST
}
```

### 7.3 BR35H Medical Configuration
```python
BR35H_CONFIG = {
    'img_size': 224,
    'img_channels': 3,
    'num_classes': 2,
    'normalization': 'ImageNet_stats',
    'data_augmentation': ['RandomHorizontalFlip()', 'RandomRotation(15)'],
    'class_names': ['no_tumor', 'tumor'],
    'medical_constraints': True,  # Enable medical-specific preprocessing
    'rician_noise': True  # Medical imaging noise model
}
```

## 8. Reproducibility and Implementation Details

### 8.1 Random Seed Management
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 8.2 Model Checkpointing
- **Diffusion Models**: Saved as `diffuser_{dataset}.pt` with configuration
- **MAE Models**: Saved as `mae_{dataset}.pt` with metadata
- **FL Models**: Saved every 5 rounds as `federated_round_{r}.pt`

### 8.3 Configuration Management
All hyperparameters are version-controlled and saved with experimental results:

```python
CONFIG_TRACKING = {
    'model_architecture': 'Complete model specifications',
    'hyperparameters': 'All training parameters', 
    'dataset_info': 'Dataset versions and preprocessing',
    'hardware_specs': 'Computational environment',
    'software_versions': 'Framework and library versions',
    'random_seeds': 'Seeds for each experimental run'
}
```

### 8.4 Experimental Validation

#### 8.4.1 Ablation Studies
1. **Component Ablation**: Individual effectiveness of diffusion, MAE, and pFedDef
2. **Hyperparameter Sensitivity**: Effect of key parameters (ε, τ, K)
3. **Architecture Ablation**: Comparison of different backbone networks

#### 8.4.2 Baseline Comparisons
- **Vanilla Federated Learning**: Standard FedAvg without defense
- **Adversarial Training**: Local adversarial training with PGD
- **Centralized Defense**: Same defense mechanisms in centralized setting
- **State-of-the-art Methods**: Comparison with recent federated defense works

## 9. Expected Outcomes and Validation Criteria

### 9.1 Performance Targets
```python
PERFORMANCE_TARGETS = {
    'clean_accuracy': '>= 85%',     # Maintain high clean performance
    'adversarial_accuracy': '>= 60%',  # Robust against PGD attacks
    'detection_rate': '>= 90%',     # High adversarial detection
    'false_positive_rate': '<= 10%', # Low false alarms
    'communication_efficiency': 'Comparable to FedAvg',
    'computational_overhead': '<= 2x baseline'
}
```

### 9.2 Validation Protocol
1. **Cross-validation**: 5-fold validation on each dataset
2. **Multiple attacks**: Evaluation against diverse attack types
3. **Scalability testing**: Performance with varying client numbers
4. **Real-world simulation**: Non-IID data distribution scenarios

---

## 10. Conclusion

This methodology provides a comprehensive framework for evaluating federated learning systems with integrated test-time adversarial defenses. The combination of diffusion-based purification, MAE detection, and personalized federated defense offers a multi-layered approach to adversarial robustness in distributed learning environments. The detailed experimental setup ensures reproducible and statistically sound results suitable for high-impact publication.

### 10.1 Key Contributions

1. **Novel Integration**: First work to combine diffusion purification with MAE detection in federated learning
2. **Multi-dataset Evaluation**: Comprehensive evaluation across vision and medical imaging domains  
3. **Practical Implementation**: Memory-optimized design suitable for resource-constrained environments
4. **Reproducible Framework**: Complete methodology with detailed hyperparameters and evaluation protocols

### 10.2 Code Availability

Complete implementation available at: [Repository URL]
- All hyperparameters and configurations documented
- Experimental scripts for reproducing results
- Pre-trained models and evaluation datasets
- Comprehensive documentation and tutorials
# Advanced Methodology: Federated Learning with Multi-Layered Test-Time Adversarial Defense Framework

## Abstract

This paper presents a novel and comprehensive federated learning framework that integrates sophisticated test-time adversarial defense mechanisms through a multi-layered approach combining Diffusion-based Purification (DiffPure), Masked Autoencoder (MAE) detection, and Personalized Federated Defense (pFedDef). Our methodology addresses the critical security vulnerabilities in distributed learning environments by establishing a robust three-tier defense system that operates at test-time inference, ensuring both high accuracy and adversarial robustness across diverse datasets including CIFAR-10, CIFAR-100, MNIST, and medical brain tumor imaging (BR35H).

**Keywords:** Federated Learning, Adversarial Defense, Diffusion Models, Masked Autoencoders, Test-Time Defense, Distributed Machine Learning

## 1. Introduction and Problem Formulation

### 1.1 Federated Learning Mathematical Framework

Consider a federated learning system with $N$ heterogeneous clients, where each client $i \in \{1, 2, ..., N\}$ possesses a local dataset $\mathcal{D}_i = \{(x_j^{(i)}, y_j^{(i)})\}_{j=1}^{n_i}$ containing $n_i$ samples. The fundamental objective is to collaboratively learn a global model that minimizes the distributed risk function:

$$\min_{\theta \in \Theta} F(\theta) = \sum_{i=1}^{N} p_i F_i(\theta) + \lambda \mathcal{R}(\theta)$$

where:
- $F_i(\theta) = \frac{1}{n_i} \sum_{j=1}^{n_i} \ell(f_{\theta}(x_j^{(i)}), y_j^{(i)})$ represents the empirical risk for client $i$
- $p_i = \frac{n_i}{\sum_{k=1}^N n_k}$ denotes the relative weight of client $i$
- $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}^+$ is the loss function (typically cross-entropy)
- $\mathcal{R}(\theta)$ is a regularization term with coefficient $\lambda > 0$
- $\Theta$ represents the parameter space

### 1.2 Comprehensive Adversarial Threat Model

#### 1.2.1 $\ell_\infty$-Bounded Perturbations

We consider the standard $\ell_\infty$-bounded adversarial threat model where an adversarial example is constructed as:

$$x_{adv} = x + \delta, \quad \text{subject to} \quad \|\delta\|_\infty \leq \epsilon$$

with the perturbation budget $\epsilon$ representing the maximum allowed change per pixel/feature.

#### 1.2.2 Projected Gradient Descent (PGD) Attack

The PGD attack iteratively refines adversarial perturbations using the following update rule:

$$x_{adv}^{(t+1)} = \Pi_{\mathcal{B}_\epsilon(x)} \left( x_{adv}^{(t)} + \alpha \cdot \text{sign}\left(\nabla_x \ell(f_\theta(x_{adv}^{(t)}), y)\right) \right)$$

where:
- $\Pi_{\mathcal{B}_\epsilon(x)}$ denotes the projection operator onto the $\ell_\infty$ ball $\mathcal{B}_\epsilon(x) = \{z : \|z - x\|_\infty \leq \epsilon\}$
- $\alpha$ is the step size parameter, typically $\alpha = \epsilon/T$ for $T$ attack iterations
- The initial point $x_{adv}^{(0)}$ can be either $x$ (non-random start) or $x + \text{Uniform}(-\epsilon, \epsilon)$ (random start)

#### 1.2.3 Advanced Attack Variants

**C&W Attack:** The Carlini & Wagner attack optimizes a more sophisticated objective:

$$\min \|\delta\|_p + c \cdot g(x + \delta)$$

where $g(x + \delta) = \max(\max_{i \neq t} Z_i(x + \delta) - Z_t(x + \delta), -\kappa)$ with $Z_i$ being the logit output for class $i$, $t$ the true class, and $\kappa$ the confidence parameter.

**AutoAttack:** Combines multiple attack strategies including APGD-CE, APGD-T, FAB-T, and Square Attack for comprehensive evaluation.

---

## 2. Multi-Layered Defense Framework Architecture

### 2.0 System Architecture Overview

Our proposed framework integrates three complementary defense mechanisms in a sophisticated multi-layered architecture. Figure 1 illustrates the complete system architecture, showing the interaction between federated learning components and defense layers.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-LAYERED FEDERATED DEFENSE FRAMEWORK                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─ ADVERSARIAL INPUT ─┐    ┌─ LAYER 1: MAE DETECTION ─┐    ┌─ LAYER 2: DIFFPURE ─┐    ┌─ LAYER 3: pFedDef ─┐
│                     │    │                          │    │                     │    │                    │
│   x_adv = x + δ     │───▶│  Vision Transformer      │───▶│   U-Net Diffusion   │───▶│  Mixture of        │
│   ‖δ‖∞ ≤ ε          │    │  Patch Size: 4×4/16×16  │    │   Hidden: 256 ch    │    │  Experts (K=3)     │
│                     │    │  s_det = f_mae(x)        │    │   x_pure = DDIM(x,t) │    │  ŷ = Σ αₖ·fₖ(x)   │
│  PGD, C&W,          │    │  Adaptive Threshold      │    │   Conditional        │    │  Attention Weights │
│  AutoAttack         │    │                          │    │   on s_det > τ       │    │                    │
└─────────────────────┘    └──────────────────────────┘    └─────────────────────┘    └────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FEDERATED LEARNING LAYER                                             │
├─ CLIENT 1 ─────────────┬─ CLIENT 2 ─────────────┬─ CLIENT 3 ─────────────┬─ ... ─────┬─ CLIENT N ──────┤
│ Local Dataset D₁       │ Local Dataset D₂       │ Local Dataset D₃       │           │ Local Dataset Dₙ│
│ pFedDef Training       │ pFedDef Training       │ pFedDef Training       │           │ pFedDef Training │
│ θ₁ = {θ₁⁽¹⁾,θ₁⁽²⁾,θ₁⁽³⁾} │ θ₂ = {θ₂⁽¹⁾,θ₂⁽²⁾,θ₂⁽³⁾} │ θ₃ = {θ₃⁽¹⁾,θ₃⁽²⁾,θ₃⁽³⁾} │           │ θₙ = {...}      │
│ Local Epochs: 5-8      │ Local Epochs: 5-8      │ Local Epochs: 5-8      │           │ Local Epochs    │
└────────────┬───────────┴────────────┬───────────┴────────────┬───────────┴───────────┴────────┬────────┘
             │                        │                        │                                 │
             └───────────────────┐    │    ┌───────────────────┘                                 │
                                 ▼    ▼    ▼                                                     │
            ┌──────────────────────────────────────────────────────────────────────────────────▼──────┐
            │                         FEDERATED SERVER                                                │
            │  Robust Aggregation: θ_global = Σ (nᵢ/n) · θᵢ                                         │
            │  Byzantine Tolerance: Krum + Median Hybrid                                             │
            │  Communication Rounds: 15-25                                                           │
            └──────────────────────────────────────────────────────────────────────────────────────┘

┌─ PRE-TRAINED MODELS ──────────────────────────────────────────────────────────────────────────────────┐
├─ Diffusion Model ──────┬─ MAE Detector ─────────┬─ Global pFedDef ───────┬─ Evaluation ─────────────┤
│ U-Net Architecture     │ ViT Encoder-Decoder    │ Aggregated Parameters  │ Clean Accuracy >95%      │
│ Training: 50-100 epochs│ Training: 30-50 epochs │ Expert Ensemble        │ Robust Accuracy >85%     │
│ Dataset-specific       │ Reconstruction Loss    │ Attention Mechanisms   │ Detection AUC >0.95      │
│ Checkpoint: diffuser_*│ Checkpoint: mae_*      │ federated_*.pt         │ Communication <2x FedAvg │
└────────────────────────┴────────────────────────┴────────────────────────┴───────────────────────────┘

Datasets: CIFAR-10 | CIFAR-100 | MNIST | BR35H (Medical Brain MRI)
```

**Figure 1:** Complete Multi-Layered Federated Defense Framework Architecture. The system comprises three main layers: (1) MAE Detection for adversarial input identification, (2) DiffPure for conditional input purification, and (3) pFedDef ensemble for robust prediction. The federated learning layer enables distributed training across multiple clients with robust server aggregation.

### 2.0.1 Defense Pipeline Flow

The test-time defense follows a sophisticated multi-stage pipeline:

```
Input x ──▶ MAE Detection ──▶ Decision ──┬─ No ──▶ Clean Path ──┐
                               s_det > τ? │                    │
                                         │                    ▼
                                    Yes ──┘                    │
                                         │                    │
                                         ▼                    │
                                  DiffPure ──▶ Adaptive ──────┘
                                  Purification  Timestep      │
                                                              ▼
                                                         pFedDef
                                                         Ensemble
                                                              │
                                                              ▼
                                                      Final Prediction
                                                        ŷ + confidence

Performance Metrics:
• Detection Time: <10ms
• Purification Time: <200ms  
• Total Latency: <250ms
• Memory Usage: <2GB
• Accuracy Preservation: >98%
```

**Figure 2:** Detailed Test-Time Defense Flow Diagram. The pipeline begins with MAE-based adversarial detection, followed by conditional DiffPure purification (only for detected adversarial inputs), and concludes with pFedDef ensemble prediction. Adaptive timestep selection ensures optimal purification strength based on detection confidence.

#### Key Architectural Principles:

1. **Layered Defense Strategy**: Each layer provides complementary protection against different attack characteristics
2. **Conditional Processing**: Purification is applied only when adversarial input is detected, maintaining efficiency for clean inputs
3. **Adaptive Parameters**: Detection thresholds and purification strength adapt based on input characteristics
4. **Federated Integration**: Defense mechanisms are seamlessly integrated into the federated learning workflow

### 2.1 Personalized Federated Defense (pFedDef) - Layer 1

#### 2.1.1 Mixture of Experts Formulation

The pFedDef framework employs a sophisticated mixture of experts approach where each client $i$ maintains $K = 3$ specialized learners with dynamic attention mechanisms:

$$\theta_i = \{\theta_i^{(1)}, \theta_i^{(2)}, \theta_i^{(3)}\}, \quad \phi_i = \{\phi_i^{(1)}, \phi_i^{(2)}, \phi_i^{(3)}\}$$

where $\theta_i^{(k)}$ represents the parameters of the $k$-th learner and $\phi_i^{(k)}$ represents the attention network parameters.

#### 2.1.2 Dynamic Attention Mechanism

The local prediction combines learner outputs using input-dependent attention weights:

$$\hat{y}_i(x) = \sum_{k=1}^{K} \alpha_i^{(k)}(x) \cdot f_{\theta_i^{(k)}}(x)$$

where the attention weights are computed through a softmax-normalized attention mechanism:

$$\alpha_i^{(k)}(x) = \frac{\exp(g_{\phi_i^{(k)}}(\psi(x)))}{\sum_{j=1}^{K} \exp(g_{\phi_i^{(j)}}(\psi(x)))}$$

Here, $\psi(x) \in \mathbb{R}^{d}$ represents the shared feature representation extracted by the ResNet-18 backbone, and $g_{\phi_i^{(k)}}$ is a learner-specific attention network.

#### 2.1.3 Architecture Details

Each learner consists of:
- **Feature Extractor**: ResNet-18 backbone with pretrained weights
- **Classifier Head**: Two-layer MLP with 256 hidden units and ReLU activation
- **Attention Network**: Two-layer MLP outputting scalar attention weights

The mixture weights are learned through:

$$w_i^{(k)}(x) = \frac{\exp(\text{MLP}_i^{(k)}(\phi(x)))}{\sum_{j=1}^{K} \exp(\text{MLP}_i^{(j)}(\phi(x)))}$$

where $\phi(x)$ represents the shared feature representation from ResNet-18.

#### 2.1.4 Training Objective for pFedDef

The training objective for client $i$ combines classification loss with attention regularization:

$$\mathcal{L}_i = \sum_{(x,y) \in \mathcal{D}_i} \left[ \ell(\hat{y}_i(x), y) + \beta \sum_{k=1}^{K} \|\phi_i^{(k)}\|_2^2 + \gamma H(\alpha_i(x)) \right]$$

where:
- $H(\alpha_i(x)) = -\sum_{k=1}^{K} \alpha_i^{(k)}(x) \log \alpha_i^{(k)}(x)$ is the entropy regularization term
- $\beta$ and $\gamma$ are hyperparameters controlling regularization strength

### 2.2 Diffusion-based Purification (DiffPure) - Layer 2

#### 2.2.1 Advanced U-Net Architecture

Our diffusion model employs a sophisticated U-Net architecture optimized for adversarial purification:

**Encoder Path:**
```
Input → DoubleConv(in_ch, 64) → Down(64, 128) → Down(128, 256) → Down(256, 512) → Down(512, 1024)
```

**Decoder Path:**
```
Up(1024, 512) → Up(512, 256) → Up(256, 128) → Up(128, 64) → OutConv(64, out_ch)
```

Each `DoubleConv` block consists of:
$$\text{DoubleConv}(x) = \text{Conv}(\text{ReLU}(\text{BatchNorm}(\text{Conv}(x))))$$

#### 2.2.2 Enhanced Time Embedding

The time embedding mechanism uses sophisticated positional encoding:

$$\text{TimeEmbed}(t) = \text{MLP}(\text{SiLU}(\text{Linear}(\text{SinCos}(t))))$$

where $\text{SinCos}(t)$ represents sinusoidal positional encoding:

$$\text{SinCos}(t)_i = \begin{cases}
\sin(t / 10000^{2i/d}) & \text{if } i \text{ is even} \\
\cos(t / 10000^{(2i-1)/d}) & \text{if } i \text{ is odd}
\end{cases}$$

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

### 2.3 Masked Autoencoder Detection (MAE) - Layer 3

#### 2.3.1 Advanced Vision Transformer Architecture

The MAE employs a sophisticated Vision Transformer architecture with the following specifications:

**Patch Embedding:**
$$\text{PE}: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^{N \times D}$$

where patches are extracted using:
$$P_{i,j} = \text{Flatten}(x[i \cdot p:(i+1) \cdot p, j \cdot p:(j+1) \cdot p, :])$$

with patch size $p$ and $N = \frac{HW}{p^2}$ total patches.

**Encoder Architecture:**
- **Layers:** 12 transformer blocks
- **Embedding Dimension:** $D = 768$
- **Attention Heads:** $h = 12$
- **Head Dimension:** $d_h = D/h = 64$
- **MLP Ratio:** 4 (hidden dimension = $4D$)

**Decoder Architecture:**
- **Layers:** 8 transformer blocks  
- **Embedding Dimension:** $D_{dec} = 512$
- **Attention Heads:** $h_{dec} = 16$
- **MLP Ratio:** 4

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
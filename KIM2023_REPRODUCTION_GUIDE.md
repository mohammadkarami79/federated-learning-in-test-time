# Kim et al., 2023 Reproduction Guide

## 📋 Overview

This guide provides instructions for reproducing the exact experimental setup from **Kim et al., 2023 - "Characterizing Internal Evasion Attacks in Federated Learning"**.

## 🎯 Key Differences from log8.txt

| Parameter | log8.txt | Kim et al., 2023 |
|-----------|----------|------------------|
| **Attack Norm** | L∞ | **L2** |
| **Epsilon (ε)** | 0.031 | **4.5** |
| **Alpha (α)** | 0.007 | **0.01** |
| **Steps (K)** | 10 | 10 (same) |
| **Architecture** | ResNet18 | **MobileNetV2** |
| **Clients** | 10 | **40** |
| **Data Split** | IID | **Non-IID (β=0.4)** |

## 🚀 Quick Start

### Method 1: Using Runner Script (Recommended)

```bash
# Run in foreground
python run_kim2023_reproduction.py

# Run in background
python run_kim2023_reproduction.py --background

# Monitor background run
tail -f kim2023_reproduction_full_*.log
```

### Method 2: Direct Execution

```bash
# Run the main script directly
python main_kim2023_reproduction.py --output-dir ./kim2023_results

# Background execution
nohup python main_kim2023_reproduction.py --output-dir ./kim2023_results > kim2023.log 2>&1 &
```

## 📊 Expected Results

Based on Kim et al., 2023 paper:

- **Test Accuracy (Clean):** ~85-90%
- **Internal Adversarial Accuracy:** ~48% (significantly better than 20% reported for standard FL)
- **Detection Rate:** Variable (depends on MAE threshold)

## 📁 File Structure

```
kim2023_results_*/
├── config.json                 # Experiment configuration
├── final_results.json          # Complete results
├── results_round_*.json        # Intermediate results
└── logs/                       # Training logs
```

## 🔧 Configuration Details

### Attack Configuration (L2-PGD)
```python
ATTACK_NORM = 'l2'               # L2 norm (NOT L∞)
ATTACK_EPSILON = 4.5             # ε = 4.5 (NOT 0.031)
ATTACK_ALPHA = 0.01              # α = 0.01 (NOT 0.007)
ATTACK_STEPS = 10                # K = 10 steps
ATTACK_TARGETED = False          # Untargeted attack
```

### Federated Learning Setup
```python
NUM_CLIENTS = 40                 # 40 clients (NOT 10)
CLIENTS_PER_ROUND = 40           # All clients participate
NON_IID = True                   # Non-IID data distribution
DIRICHLET_BETA = 0.4             # Dirichlet parameter
```

### Architecture
```python
ARCHITECTURE = 'mobilenetv2'     # MobileNetV2 (NOT ResNet18)
```

## 🛡️ Defense Components

The experiment uses the same defense components as `main.py`:

1. **MAE Detector:** Pre-trained CIFAR-10 MAE for adversarial detection
2. **DiffPure:** Pre-trained diffusion model for input purification  
3. **pFedDef:** Personalized federated defense (simulated via ensemble)

## 📈 Monitoring Progress

### Real-time Monitoring
```bash
# Watch log file
tail -f kim2023_reproduction_*.log

# Monitor specific metrics
grep -E "Round.*Clean Acc|Adv Acc" kim2023_reproduction_*.log
```

### Check Results
```bash
# View final results
cat kim2023_results_*/final_results.json | jq '.final_metrics'

# Compare with log8.txt baseline
echo "Kim2023 vs log8.txt comparison:"
echo "Architecture: MobileNetV2 vs ResNet18" 
echo "Attack: L2-PGD vs L∞-PGD"
echo "Clients: 40 vs 10"
```

## 🔍 Key Metrics to Watch

1. **Clean Accuracy:** Should reach 85-90% (similar to log8.txt)
2. **L2-Adversarial Accuracy:** Target ~48% (much better than L∞)
3. **Detection Rate:** MAE detection of L2 adversarial examples
4. **Training Time:** Longer due to 40 clients vs 10

## 🚨 Troubleshooting

### Missing Models
```bash
# Check for required checkpoints
ls -la checkpoints/diffuser_cifar10.pt
ls -la checkpoints/mae_detector_cifar10.pt

# If missing, they should be available from previous runs
# Or train them using main.py first
```

### Memory Issues
```bash
# Reduce batch size in config if needed
# 40 clients require more memory than 10
```

### Slow Training
```bash
# Expected: ~4x longer than log8.txt due to 40 clients
# Each round: 40 clients × training time per client
```

## 📚 References

- **Original Paper:** Kim et al., 2023 - "Characterizing Internal Evasion Attacks in Federated Learning"
- **Code Repository:** [pFedDef_v1](https://github.com/tjkim/pFedDef_v1)
- **Baseline Comparison:** log8.txt (our previous CIFAR-10 experiment)

## ✅ Verification Checklist

- [ ] L2-norm PGD attack (ε=4.5, α=0.01, K=10)
- [ ] MobileNetV2 architecture  
- [ ] 40 clients with Non-IID data (β=0.4)
- [ ] Defense components loaded (MAE + DiffPure)
- [ ] Results comparable to Kim et al., 2023 benchmarks

## 🎯 Success Criteria

**Successful reproduction if:**
1. ✅ Clean accuracy: 85-90%
2. ✅ L2-adversarial accuracy: 45-50% (vs ~20% without defense)
3. ✅ Experiment completes without errors
4. ✅ Results saved in structured JSON format

---

**Note:** This experiment preserves `main.py` completely unchanged for reproducibility of previous results.

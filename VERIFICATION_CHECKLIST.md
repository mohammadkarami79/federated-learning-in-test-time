# 🔍 Configuration Verification Checklist

## To ensure you get the same results as log7.txt (87.7% clean, 72.1% adversarial accuracy)

### ✅ **Critical Parameters (Must Match log7.txt)**

| Parameter | Expected Value | Your Value | Status |
|-----------|----------------|------------|--------|
| `DATASET` | 'cifar10' | | |
| `NUM_CLASSES` | 10 | | |
| `NUM_CLIENTS` | 10 | | |
| `NUM_ROUNDS` | 15 | | |
| `CLIENT_EPOCHS` | 8 | | |
| `BATCH_SIZE` | 64 | | |
| `LEARNING_RATE` | 0.01 | | |
| `MAE_THRESHOLD` | 0.15 | | |
| `DIFFUSER_STEPS` | 4 | | |
| `DIFFUSER_SIGMA` | 0.3 | | |
| `ATTACK_EPSILON` | 0.031 | | |
| `ATTACK_STEPS` | 10 | | |

### ✅ **Required Attributes (Must Be Present)**

- [ ] `DATA_ROOT` = 'data'
- [ ] `DATA_PATH` = 'data' 
- [ ] `MODE` = 'full'
- [ ] `DATA_DISTRIBUTION` = 'iid'
- [ ] `EVAL_BATCH_SIZE` = 128
- [ ] `ENABLE_MAE_DETECTOR` = True
- [ ] `ENABLE_DIFFPURE` = True
- [ ] `SELECTIVE_DEFENSE` = True

### ✅ **File Structure Check**

- [ ] `config_selective_defense.py` exists and is updated
- [ ] `run_selective_defense.py` exists and is updated
- [ ] `main.py` exists and has `run_federated_training` function
- [ ] `utils/data_utils.py` exists
- [ ] `defense/mae_detector.py` exists
- [ ] `diffusion/diffuser.py` exists
- [ ] `attacks/pgd.py` exists

### ✅ **Dependencies Check**

- [ ] PyTorch installed
- [ ] Torchvision installed
- [ ] CUDA available (if using GPU)
- [ ] All required Python packages installed

### ✅ **Quick Test Commands**

Run these commands to verify everything works:

```bash
# 1. Test configuration loading
python -c "from config_selective_defense import get_config; print('Config OK:', len(get_config()))"

# 2. Test data loading
python -c "from config_selective_defense import get_config; from utils.data_utils import get_dataset; import types; cfg = types.SimpleNamespace(**get_config()); train, test = get_dataset(cfg); print('Data OK:', len(train), len(test))"

# 3. Test model creation
python -c "import torchvision.models as models; import torch.nn as nn; model = models.resnet18(pretrained=False); model.fc = nn.Linear(model.fc.in_features, 10); print('Model OK:', sum(p.numel() for p in model.parameters()))"
```

### ✅ **Expected Training Output**

When you run training, you should see:

1. **Round 1**: Clean Acc: ~47%, Adv Acc: ~27%
2. **Round 5**: Clean Acc: ~83%, Adv Acc: ~67%
3. **Round 10**: Clean Acc: ~86%, Adv Acc: ~70%
4. **Round 15**: Clean Acc: ~87%, Adv Acc: ~72%

### ✅ **Success Indicators**

- [ ] No "AttributeError" or missing attribute errors
- [ ] Dataset loads successfully (50,000 train, 10,000 test)
- [ ] MAE detection rate stays around 15.6%
- [ ] Training progresses through all 15 rounds
- [ ] Final clean accuracy reaches 85%+
- [ ] Final adversarial accuracy reaches 70%+

### 🚀 **Ready to Run**

Once all checks pass:

```bash
nohup python run_selective_defense.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 📊 **Monitoring Progress**

```bash
# Watch the log
tail -f training_*.log

# Check for errors
grep -i error training_*.log

# Check progress
grep "Round.*Clean Acc" training_*.log
```

### 🎯 **Expected Final Results**

- **Clean Accuracy**: 87.7% ± 1%
- **Adversarial Accuracy**: 72.1% ± 1%
- **MAE Detection Rate**: 15.6% ± 1%
- **Training Time**: ~12 hours
- **Total Rounds**: 15

If you get these results, your configuration is perfect and matches log7.txt!

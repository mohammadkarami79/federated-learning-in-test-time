#!/bin/bash
# =============================================================================
# MedFedPure – Server Execution Script
# =============================================================================
# Run this file on the Linux server step-by-step.
# All commands are designed to run from the PROJECT ROOT directory.
#
# STEP 0: Navigate to project root
#   cd /path/to/federated-learning-in-test-time
#
# =============================================================================

set -euo pipefail

# ── Configuration (edit these before running) ─────────────────────────────────
DATASET="br35h"            # Primary dataset: br35h or cifar10
SEEDS="42 43 44"           # Space-separated seeds (3 recommended)
ADDL_DATASET="breastmnist" # Additional medical dataset (or brats)
ADDL_IMG_SIZE=64           # 64 for MedMNIST, 224 for BraTS
OUT_DIR="experiment_results"
NUM_ROUNDS=20              # Communication rounds (paper: 20)
LOCAL_EPOCHS=15            # Local epochs per round (paper: 15)
NUM_CLIENTS=10             # Number of FL clients (paper: 10)

# ── Environment check ─────────────────────────────────────────────────────────
echo "========================================================"
echo "MedFedPure – Experiment Runner"
echo "========================================================"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""

# ── Install additional dependencies ──────────────────────────────────────────
echo "[Setup] Installing medmnist for additional dataset experiments..."
pip install medmnist --quiet

# ── PART 1: Ablation Study on Br35H ─────────────────────────────────────────
echo ""
echo "========================================================"
echo "PART 1 – Ablation Study on ${DATASET^^}"
echo "========================================================"
python3 experiments/run_all_experiments.py \
    --dataset "${DATASET}" \
    --data-root data \
    --seeds ${SEEDS} \
    --parts ablation \
    --rounds ${NUM_ROUNDS} \
    --epochs ${LOCAL_EPOCHS} \
    --clients ${NUM_CLIENTS} \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee logs/ablation_${DATASET}.log

echo "Part 1 complete. Results in: ${OUT_DIR}/ablation/"

# ── PART 2: Additional Medical Dataset ───────────────────────────────────────
echo ""
echo "========================================================"
echo "PART 2 – Additional Dataset: ${ADDL_DATASET^^}"
echo "========================================================"
python3 experiments/run_all_experiments.py \
    --dataset "${DATASET}" \
    --data-root data \
    --seeds ${SEEDS} \
    --parts additional \
    --additional-dataset "${ADDL_DATASET}" \
    --additional-img-size ${ADDL_IMG_SIZE} \
    --rounds ${NUM_ROUNDS} \
    --epochs ${LOCAL_EPOCHS} \
    --clients ${NUM_CLIENTS} \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee logs/additional_${ADDL_DATASET}.log

echo "Part 2 complete. Results in: ${OUT_DIR}/additional_dataset/"

# ── PART 3: Baseline Comparison ──────────────────────────────────────────────
echo ""
echo "========================================================"
echo "PART 3 – Baseline Comparison on ${DATASET^^}"
echo "========================================================"
python3 experiments/run_all_experiments.py \
    --dataset "${DATASET}" \
    --data-root data \
    --seeds ${SEEDS} \
    --parts baseline \
    --rounds ${NUM_ROUNDS} \
    --epochs ${LOCAL_EPOCHS} \
    --clients ${NUM_CLIENTS} \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee logs/baseline_${DATASET}.log

echo "Part 3 complete. Results in: ${OUT_DIR}/baseline_comparison/"

# ── PART 4: Efficiency Analysis ───────────────────────────────────────────────
echo ""
echo "========================================================"
echo "PART 4 – Efficiency Analysis on ${DATASET^^}"
echo "========================================================"
python3 experiments/run_all_experiments.py \
    --dataset "${DATASET}" \
    --data-root data \
    --seeds 42 \
    --parts efficiency \
    --rounds ${NUM_ROUNDS} \
    --epochs ${LOCAL_EPOCHS} \
    --clients ${NUM_CLIENTS} \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee logs/efficiency_${DATASET}.log

echo "Part 4 complete. Results in: ${OUT_DIR}/efficiency/"

# ── Final Summary ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Collect these files and send to the team:"
echo ""
echo "  ${OUT_DIR}/ablation/ablation_${DATASET}_*.csv"
echo "  ${OUT_DIR}/ablation/ablation_${DATASET}_*_summary.txt"
echo "  ${OUT_DIR}/ablation/aggregated_${DATASET}_aggregated.json"
echo "  ${OUT_DIR}/additional_dataset/additional_${ADDL_DATASET}_*.csv"
echo "  ${OUT_DIR}/baseline_comparison/baseline_comparison_${DATASET}_*.csv"
echo "  ${OUT_DIR}/efficiency/efficiency_${DATASET}_*.csv"
echo "  ${OUT_DIR}/experiment_summary.json"
echo "  logs/ablation_${DATASET}.log"
echo "  logs/additional_${ADDL_DATASET}.log"
echo "  logs/baseline_${DATASET}.log"
echo "  logs/efficiency_${DATASET}.log"
echo "========================================================"

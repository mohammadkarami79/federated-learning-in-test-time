#!/bin/bash

# Complete Federated Learning Experiments Runner
# This script runs all experiments for all datasets

set -e  # Exit on any error

echo "🚀 Starting Complete Federated Learning Experiments..."

# Set CUDA environment variable
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Activate virtual environment
source venv/bin/activate

# Function to train MAE and Diffusion for a dataset
train_models() {
    local dataset=$1
    echo "📚 Training models for dataset: $dataset"
    
    # Train MAE Detector
    echo "🔍 Training MAE Detector for $dataset..."
    python scripts/train_mae_detector.py --dataset $dataset --epochs 30 --batch-size 64
    
    # Rename checkpoint
    mv checkpoints/mae_detector.pt checkpoints/mae_detector_${dataset}.pt
    
    # Train Diffusion Model
    echo "🎨 Training Diffusion Model for $dataset..."
    python train_diffpure.py --dataset $dataset --epochs 50 --batch-size 64 --hidden-channels 256
    
    echo "✅ Models trained for $dataset"
}

# Function to run main experiment
run_experiment() {
    local dataset=$1
    echo "🚀 Running main experiment for $dataset..."
    
    python main.py --dataset $dataset --mode full --train-diffusion --skip-setup > logs_${dataset}.txt 2>&1
    
    echo "✅ Experiment completed for $dataset"
}

# Train models for all datasets
echo "🔄 Training MAE and Diffusion models for all datasets..."
train_models "br35h"
train_models "cifar10"
train_models "cifar100"
train_models "mnist"

# Run main experiments for all datasets
echo "🔄 Running main experiments for all datasets..."
run_experiment "br35h"
run_experiment "cifar10"
run_experiment "cifar100"
run_experiment "mnist"

# Collect results
echo "📊 Collecting all results..."
python paper_results_collector.py

echo "🎉 All experiments completed successfully!"
echo "📁 Results saved in paper_results/"
echo "📝 Logs saved in logs_*.txt files"

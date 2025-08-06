#!/usr/bin/env python3
"""
Paper Results Collector
Collects and organizes all training results in a specific folder for paper preparation
"""

import os
import shutil
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def create_paper_results_folder():
    """Create a specific folder for paper results"""
    paper_folder = Path("paper_results")
    paper_folder.mkdir(exist_ok=True)
    
    # Create subfolders
    (paper_folder / "tables").mkdir(exist_ok=True)
    (paper_folder / "figures").mkdir(exist_ok=True)
    (paper_folder / "logs").mkdir(exist_ok=True)
    (paper_folder / "checkpoints").mkdir(exist_ok=True)
    (paper_folder / "raw_data").mkdir(exist_ok=True)
    
    return paper_folder

def collect_results_for_dataset(dataset_name, paper_folder):
    """Collect all results for a specific dataset"""
    logger = logging.getLogger(__name__)
    
    # Create dataset-specific folder
    dataset_folder = paper_folder / f"dataset_{dataset_name}"
    dataset_folder.mkdir(exist_ok=True)
    
    # Copy analysis results
    analysis_source = Path("analysis_results")
    if analysis_source.exists():
        for file in analysis_source.glob("*"):
            if dataset_name.lower() in file.name.lower():
                shutil.copy2(file, dataset_folder / file.name)
                logger.info(f"✅ Copied {file.name} to {dataset_folder}")
    
    # Copy checkpoints
    checkpoint_source = Path("checkpoints")
    if checkpoint_source.exists():
        for file in checkpoint_source.glob(f"*{dataset_name}*"):
            shutil.copy2(file, dataset_folder / file.name)
            logger.info(f"✅ Copied checkpoint {file.name}")
    
    # Copy training logs
    log_files = ["training.log", "main.log"]
    for log_file in log_files:
        if Path(log_file).exists():
            shutil.copy2(log_file, dataset_folder / f"{dataset_name}_{log_file}")
            logger.info(f"✅ Copied log {log_file}")
    
    return dataset_folder

def create_results_summary(paper_folder):
    """Create a summary of all collected results"""
    logger = logging.getLogger(__name__)
    
    summary = {
        "collection_date": datetime.now().isoformat(),
        "datasets": [],
        "overall_status": "incomplete"
    }
    
    # Check which datasets have results
    for dataset_folder in paper_folder.glob("dataset_*"):
        dataset_name = dataset_folder.name.replace("dataset_", "")
        
        dataset_info = {
            "name": dataset_name,
            "has_diffusion": False,
            "has_mae": False,
            "has_results": False,
            "files": []
        }
        
        # Check for diffusion model
        diffusion_files = list(dataset_folder.glob("*diffuser*"))
        if diffusion_files:
            dataset_info["has_diffusion"] = True
            dataset_info["files"].extend([f.name for f in diffusion_files])
        
        # Check for MAE model
        mae_files = list(dataset_folder.glob("*mae*"))
        if mae_files:
            dataset_info["has_mae"] = True
            dataset_info["files"].extend([f.name for f in mae_files])
        
        # Check for results
        result_files = list(dataset_folder.glob("*summary*")) + list(dataset_folder.glob("*stats*"))
        if result_files:
            dataset_info["has_results"] = True
            dataset_info["files"].extend([f.name for f in result_files])
        
        summary["datasets"].append(dataset_info)
    
    # Save summary
    summary_file = paper_folder / "results_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Results summary saved to {summary_file}")
    return summary

def print_collection_status(summary):
    """Print the status of collected results"""
    print("\n" + "="*60)
    print("PAPER RESULTS COLLECTION STATUS")
    print("="*60)
    
    for dataset in summary["datasets"]:
        status = []
        if dataset["has_diffusion"]:
            status.append("✅ Diffusion")
        else:
            status.append("❌ Diffusion")
        
        if dataset["has_mae"]:
            status.append("✅ MAE")
        else:
            status.append("❌ MAE")
        
        if dataset["has_results"]:
            status.append("✅ Results")
        else:
            status.append("❌ Results")
        
        print(f"{dataset['name'].upper():<15} | {' | '.join(status)}")
    
    print("="*60)
    print("Next steps:")
    print("1. Run training for missing components")
    print("2. Check paper_results/ folder for collected data")
    print("3. Use results_summary.json for status tracking")

def main():
    """Main function to collect all results"""
    logger = setup_logging()
    
    print("📊 PAPER RESULTS COLLECTOR")
    print("="*60)
    
    # Create paper results folder
    paper_folder = create_paper_results_folder()
    logger.info(f"✅ Created paper results folder: {paper_folder}")
    
    # Collect results for each dataset
    datasets = ["br35h", "cifar10", "cifar100", "mnist"]
    
    for dataset in datasets:
        logger.info(f"🔄 Collecting results for {dataset}...")
        collect_results_for_dataset(dataset, paper_folder)
    
    # Create summary
    summary = create_results_summary(paper_folder)
    
    # Print status
    print_collection_status(summary)
    
    print(f"\n🎉 Results collected in: {paper_folder}")
    print("📁 Check the folder for all your paper data!")

if __name__ == "__main__":
    main() 
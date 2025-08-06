#!/usr/bin/env python3
"""
Model Quality Checker
Checks if existing models are trained with proper parameters for paper quality
"""

import torch
import json
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

def check_diffusion_model_quality(dataset_name):
    """Check if diffusion model exists and has proper quality"""
    logger = logging.getLogger(__name__)
    
    model_path = Path(f"checkpoints/diffuser_{dataset_name}.pt")
    config_path = Path(f"checkpoints/diffuser_{dataset_name}_config.json")
    
    if not model_path.exists():
        logger.warning(f"❌ No diffusion model found for {dataset_name}")
        return False, "No model file"
    
    # Check model file size (rough quality indicator)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"📁 Diffusion model size: {file_size_mb:.1f} MB")
    
    # Check if config exists and has proper parameters
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check for paper-quality parameters
            has_proper_epochs = config.get('epochs', 0) >= 10
            has_proper_steps = config.get('diffusion_steps', 0) >= 100
            
            logger.info(f"📊 Config epochs: {config.get('epochs', 'N/A')}")
            logger.info(f"📊 Config diffusion_steps: {config.get('diffusion_steps', 'N/A')}")
            
            if has_proper_epochs and has_proper_steps:
                logger.info(f"✅ Diffusion model for {dataset_name} appears to be paper-quality")
                return True, "Paper quality"
            else:
                logger.warning(f"⚠️ Diffusion model for {dataset_name} may not be paper-quality")
                return False, "Low quality parameters"
                
        except Exception as e:
            logger.warning(f"⚠️ Could not read config for {dataset_name}: {e}")
            return False, "No config or invalid config"
    else:
        logger.warning(f"⚠️ No config file for {dataset_name} - cannot verify quality")
        return False, "No config file"
    
    return True, "Unknown quality"

def check_mae_model_quality(dataset_name):
    """Check if MAE model exists and has proper quality"""
    logger = logging.getLogger(__name__)
    
    # Check for MAE model files
    mae_files = list(Path("checkpoints").glob(f"*mae*{dataset_name}*"))
    
    if not mae_files:
        logger.info(f"ℹ️ No MAE model files found for {dataset_name} - will use built-in")
        return False, "Using built-in MAE"
    
    # Check file sizes
    total_size_mb = sum(f.stat().st_size for f in mae_files) / (1024 * 1024)
    logger.info(f"📁 MAE model total size: {total_size_mb:.1f} MB")
    
    if total_size_mb > 10:  # Rough indicator of proper training
        logger.info(f"✅ MAE model for {dataset_name} appears substantial")
        return True, "Substantial model"
    else:
        logger.warning(f"⚠️ MAE model for {dataset_name} may be insufficient")
        return False, "Small model size"
    
    return True, "Unknown quality"

def check_training_results_quality(dataset_name):
    """Check if training results exist and have proper metrics"""
    logger = logging.getLogger(__name__)
    
    # Check for result files
    result_files = list(Path("analysis_results").glob(f"*{dataset_name}*"))
    
    if not result_files:
        logger.warning(f"❌ No training results found for {dataset_name}")
        return False, "No results"
    
    # Check if results have proper metrics
    for result_file in result_files:
        if result_file.suffix == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(result_file)
                
                # Check for proper columns
                required_cols = ['Clean Acc', 'Adv Acc', 'MAE Detection']
                has_proper_cols = all(col in df.columns for col in required_cols)
                
                if has_proper_cols and len(df) > 0:
                    logger.info(f"✅ Training results for {dataset_name} have proper metrics")
                    return True, "Proper metrics"
                else:
                    logger.warning(f"⚠️ Training results for {dataset_name} may be incomplete")
                    return False, "Incomplete metrics"
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not read results for {dataset_name}: {e}")
                return False, "Unreadable results"
    
    return False, "No proper results"

def main():
    """Main function to check model quality"""
    logger = setup_logging()
    
    print("🔍 MODEL QUALITY CHECKER")
    print("="*60)
    
    datasets = ["br35h", "cifar10", "cifar100", "mnist"]
    
    results = {}
    
    for dataset in datasets:
        logger.info(f"\n🔄 Checking {dataset.upper()}...")
        
        # Check diffusion model
        diffusion_ok, diffusion_reason = check_diffusion_model_quality(dataset)
        
        # Check MAE model
        mae_ok, mae_reason = check_mae_model_quality(dataset)
        
        # Check training results
        results_ok, results_reason = check_training_results_quality(dataset)
        
        results[dataset] = {
            "diffusion": {"ok": diffusion_ok, "reason": diffusion_reason},
            "mae": {"ok": mae_ok, "reason": mae_reason},
            "results": {"ok": results_ok, "reason": results_reason}
        }
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL QUALITY SUMMARY")
    print("="*60)
    
    for dataset, checks in results.items():
        print(f"\n{dataset.upper():<15}:")
        print(f"  Diffusion: {'✅' if checks['diffusion']['ok'] else '❌'} - {checks['diffusion']['reason']}")
        print(f"  MAE:       {'✅' if checks['mae']['ok'] else '❌'} - {checks['mae']['reason']}")
        print(f"  Results:   {'✅' if checks['results']['ok'] else '❌'} - {checks['results']['reason']}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    for dataset, checks in results.items():
        needs_training = []
        
        if not checks['diffusion']['ok']:
            needs_training.append("diffusion")
        if not checks['mae']['ok'] and "built-in" not in checks['mae']['reason']:
            needs_training.append("MAE")
        if not checks['results']['ok']:
            needs_training.append("pipeline")
        
        if needs_training:
            print(f"📝 {dataset.upper()}: Run training for {', '.join(needs_training)}")
            if "diffusion" in needs_training:
                print(f"   → python main.py --dataset {dataset} --mode full --train-diffusion --skip-setup")
            if "MAE" in needs_training:
                print(f"   → python main.py --dataset {dataset} --mode full --train-mae --skip-setup")
            if "pipeline" in needs_training:
                print(f"   → python main.py --dataset {dataset} --mode full --skip-setup")
        else:
            print(f"✅ {dataset.upper()}: All components appear ready")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 
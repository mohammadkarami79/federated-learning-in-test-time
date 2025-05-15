"""
Quick import test for all project modules.
This script simply tries to import every module in the project.
Used to verify that there are no import errors.
"""

import os
import sys
import importlib
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modules to test
MODULES_TO_TEST = [
    # Core modules
    'config',
    'main',
    'models',
    
    # Defense modules
    'defense.combined_defense',
    
    # Diffusion modules
    'diffusion',
    'diffusion.diffuser',
    
    # Federated modules
    'federated',
    'federated.client',
    'federated.server',
    'federated.trainer',
    
    # Attack modules
    'attacks',
    'attacks.pgd',
    'attacks.fgsm',
    'attacks.internal_pgd',
    
    # Model modules
    'models.pfeddef_model',
    'models.resnet',
    
    # Metrics modules
    'metrics',
    'metrics.logger',
    
    # Utility modules
    'utils.data_utils',
    'utils.model_manager',
]

def test_imports():
    """Test importing all modules."""
    print("Starting quick import test...")
    
    success_count = 0
    failure_count = 0
    failed_modules = []
    
    for module_name in MODULES_TO_TEST:
        try:
            print(f"  Importing {module_name}...", end="")
            module = importlib.import_module(module_name)
            print(" ✓")
            success_count += 1
        except Exception as e:
            print(f" ✗ - Error: {e}")
            traceback.print_exc()
            failure_count += 1
            failed_modules.append((module_name, str(e)))
    
    # Print summary
    print("\nImport test summary:")
    print(f"  Successful imports: {success_count}")
    print(f"  Failed imports: {failure_count}")
    
    if failed_modules:
        print("\nFailed modules:")
        for module_name, error in failed_modules:
            print(f"  • {module_name}: {error}")
        return False
    else:
        print("\nAll modules import cleanly ✅")
        return True

if __name__ == "__main__":
    success = test_imports()
    # Return appropriate exit code
    sys.exit(0 if success else 1) 
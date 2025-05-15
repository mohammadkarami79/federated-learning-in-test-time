# Configuration Coverage Audit

This document lists all configuration keys used across the codebase and verifies whether they are defined in the default `Config` class and in the `PRESETS` dictionaries.

## Key Coverage

| Config Key | In Default Config | In Debug Preset | In Full Preset | Used In Files |
|------------|------------------|-----------------|----------------|--------------|
| N_CLIENTS | ✅ | ✅ | ✅ | main.py, scripts/sanity_suite.py, federated/trainer.py, tests/test_pipeline.py, train_combined_defense.py |
| N_LEARNERS | ✅ | ✅ | ✅ | models/pfeddef_model.py, federated/server.py, federated/client.py, train_combined_defense.py |
| N_ROUNDS | ✅ | ✅ | ✅ | main.py, federated/trainer.py, train_combined_defense.py |
| LOCAL_EPOCHS | ✅ | ✅ | ✅ | main.py, federated/trainer.py, federated/server.py, train_combined_defense.py |
| BATCH_SIZE | ✅ | ✅ | ✅ | main.py, federated/client.py, tests/test_pipeline.py, utils/data_utils.py, quick_test.py, train_combined_defense.py, attacks/internal_pgd.py |
| DIFFUSER_STEPS | ✅ | ✅ | ✅ | main.py, federated/client.py, defense/combined_defense.py, quick_test.py, scripts/sanity_suite.py, train_combined_defense.py, attacks/internal_pgd.py |
| DIFFUSER_SIGMA | ✅ | ✅ | ✅ | main.py, federated/client.py, federated/trainer.py, defense/combined_defense.py, scripts/sanity_suite.py, train_combined_defense.py, attacks/internal_pgd.py |
| PGD_STEPS | ✅ | ✅ | ✅ | main.py, federated/client.py, federated/trainer.py, train_combined_defense.py |
| PGD_EPS | ✅ | ❌ | ❌ | main.py, federated/trainer.py |
| PGD_ALPHA | ✅ | ❌ | ❌ | main.py, federated/client.py |
| LEARNING_RATE | ✅ | ❌ | ❌ | federated/client.py |
| LAMBDA_KL | ✅ | ❌ | ❌ | federated/client.py |
| DIRICHLET_ALPHA | ✅ | ❌ | ❌ | utils/data_utils.py |
| LOCAL_STEPS_PER_EPOCH | ✅ | ❌ | ❌ | Not directly referenced |
| verbose | ✅ | ❌ | ❌ | federated/trainer.py |
| output_dir | ✅ | ❌ | ❌ | federated/trainer.py, train_combined_defense.py |
| model_dir | ✅ | ❌ | ❌ | tests/test_pipeline.py |
| dataset | ❌ | ❌ | ❌ | tests/test_pipeline.py |
| MAX_STEPS | ❌ | ❌ | ❌ | scripts/sanity_suite.py |
| experiment_name | ❌ | ❌ | ❌ | train_combined_defense.py |

## Missing Configuration Keys

The following keys are used in the code but missing from both the default Config class and PRESETS:

1. `dataset` - Used in tests/test_pipeline.py
2. `MAX_STEPS` - Used in scripts/sanity_suite.py (only defined locally in QuickDebugConfig)
3. `experiment_name` - Used in train_combined_defense.py (set at runtime)

## Presets Coverage Analysis

Several configuration keys are defined in the default Config class but are missing from the presets:

1. `PGD_EPS` - Missing from both presets (has default value 8/255)
2. `PGD_ALPHA` - Missing from both presets (has default value 2/255)
3. `LEARNING_RATE` - Missing from both presets (has default value 0.001)
4. `LAMBDA_KL` - Missing from both presets (has default value 1.0)
5. `DIRICHLET_ALPHA` - Missing from both presets (has default value 0.3)
6. `LOCAL_STEPS_PER_EPOCH` - Missing from both presets (has default value 100)
7. `verbose` - Missing from both presets (has default value True)
8. `output_dir` - Missing from both presets (has default value 'results')
9. `model_dir` - Missing from both presets (has default value 'models')

## Recommendations

1. Add missing keys to the PRESETS dictionaries for better consistency.
2. Add `dataset` and `experiment_name` to the default Config class.
3. Remove `MAX_STEPS` from QuickDebugConfig if it's not used or add it to the default Config. 
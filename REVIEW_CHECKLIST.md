# <font color='SlateBlue'>REVIEW_CHECKLIST, Version 1</font>

- [REVIEW\_CHECKLIST, Version 1](#review_checklist-version-1)
  - [Coding and Documentation ***Style*** Problems](#coding-and-documentation-style-problems)
    - [Coding Style Problmes](#coding-style-problmes)
  - [Coding and Documentation ***Logic*** Problems](#coding-and-documentation-logic-problems)
    - [Coding Logic Problmes](#coding-logic-problmes)
  - [Coding and Documentation ***Usage*** Problems](#coding-and-documentation-usage-problems)
    - [Coding Usage Problmes](#coding-usage-problmes)

## <font color='MediumOrchid'>Coding and Documentation ***Style*** Problems</font>

### <font color='Thistle'>Coding Style Problmes</font>

1.  [<font color='Yellow'>Partially Solved</font>] Several references to PfedDef + Diffpure within the codes, which are not the real name of our paper.
2. [<font color='Yellow'>Partially Solved</font>] Exact numbers in descriptions (e.g. 2-5 minutes for time of training), which I guess are better not to be used in code. 
3. [<font color='Red'>To be Solved</font>] Some weird stickers in logging, which I think exposes that we extensively used AI in our project, and don't look good overall. Please remove them if you are agree with me.
4. [<font color='Green'>Solved</font>] in `config_fixed.py#main` cfg.N_TASKS where never defined in any of the configs, so I replaced it with the global var N_TASKS.


## <font color='MediumOrchid'>Coding and Documentation ***Logic*** Problems</font>

### <font color='Thistle'>Coding Logic Problmes</font>

1. [<font color='Red'>To be Solved</font>] In `main.py#train_diffusion_model`, The diffusion model is being trained for only <font color='Red'>3</font> epoches, which I think this number is better to be controlled from the outside, or ***at least be increased***. <font color='Cyan'>(It MAYBE is fine tuning an already trained model, so please check `train_diffpure.py` to ensure what is happening)</font>
2. [<font color='Red'>To be Solved</font>] In `main.py#train_diffusion_model`, I think it is better to do nothing in case diffusion training is failed, because the minimal diffusion solution is not gonna help having good results, and may be misleading somehow. It may be better forcing one fix the main diffusion model, and remove the fallback.
3. [<font color='Red'>To be Solved</font>] In `main.py#train_mae_detector`, same problems of "1." and "2." for MAE. Besides, MAW doesn't even have a fallback plan! It's just printing a message about fallback, which is very misleading.
4. [<font color='Red'>To be Solved</font>] In `main.py#run_federated_training`, in case of not having the diffusion model, it is using the diffuser without training!
5. [<font color='Red'>To be Solved</font>] In `main.py#run_federated_training`, it is force reducing the number of clients to ***3*** and it is training each client for only ***1*** epochs.
6. [<font color='Yellow'>Partially Solved</font>] in `config_fixed.py#get_full_config` I thought the hyperparameters used were weak for a full training session, so I changed the full_config and logged the changes bellow the new function.

## <font color='MediumOrchid'>Coding and Documentation ***Usage*** Problems</font>

### <font color='Thistle'>Coding Usage Problmes</font>

1. [<font color='Red'>To be Solved</font>] These modules were never used, delete them if they are not needed:
   - `train_combined_defense.py` , `federated/trainer.py`: These two do something that is already being done in `main.py#run_federated_training`, but I suspect that function is as complete and correct as them. needs to be checked and validated.
   - 
2. [<font color='Red'>To be Solved</font>] Some testing modules better not be included in the final project: `test_config_integration.py` , `simple_test.py`
# <font color='SlateBlue'>REVIEW\_CHECKLIST, Version 2</font>

- [REVIEW\_CHECKLIST, Version 2](#review_checklist-version-2)
  - [Coding and Documentation ***Style*** Issues](#coding-and-documentation-style-issues)
    - [Coding Style Issues](#coding-style-issues)
  - [Coding and Documentation ***Logic*** Issues](#coding-and-documentation-logic-issues)
    - [Coding Logic Issues](#coding-logic-issues)
  - [Coding and Documentation ***Usage*** Issues](#coding-and-documentation-usage-issues)
    - [Coding Usage Issues](#coding-usage-issues)

---

## <font color='MediumOrchid'>Coding and Documentation ***Style*** Issues</font>

### <font color='Thistle'>Coding Style Issues</font>

1. \[<font color='Yellow'>Partially Resolved</font>] Several references to "PfedDef" and "Diffpure" remain in the code, though they do not reflect the actual name of our paper.
2. \[<font color='Yellow'>Partially Resolved</font>] Descriptions include specific numbers (e.g., “2–5 minutes” for training time), which may be too rigid or inaccurate. It’s better to generalize such estimates in documentation.
3. \[<font color='Red'>To be Resolved</font>] Some logging outputs contain odd emoji/sticker elements that make the project appear overly reliant on AI tools. These do not look professional—please remove them if you agree.
4. \[<font color='Green'>Resolved</font>] In `config_fixed.py#main`, `cfg.N_TASKS` was not defined in any config files. I replaced it with the global variable `N_TASKS`.

---

## <font color='MediumOrchid'>Coding and Documentation ***Logic*** Issues</font>

### <font color='Thistle'>Coding Logic Issues</font>

1. \[<font color='Red'>To be Resolved</font>] In `main.py#train_diffusion_model`, the diffusion model is trained for only <font color='Red'>3</font> epochs. This value should be configurable, or at the very least, increased.
2. \[<font color='Red'>To be Resolved</font>] In `main.py#train_diffusion_model`, if diffusion training fails, it continues with a minimal backup. This fallback may lead to poor results and confusion. It would be better to stop execution and force the user to fix the core issue.
3. \[<font color='Red'>To be Resolved</font>] In `main.py#train_mae_detector`, the same issues as in point 1 and 2 apply. In addition, MAE doesn't even have a real fallback—just a misleading log message claiming one exists.
4. \[<font color='Red'>To be Resolved</font>] In `main.py#run_federated_training`, if the diffusion model is unavailable, it proceeds using the diffuser *without* any training.
5. \[<font color='Red'>To be Resolved</font>] In `main.py#run_federated_training`, the number of clients is hard-coded to ***3***, and each client trains for only ***1*** epoch. These should be parameterized.
6. \[<font color='Yellow'>Partially Resolved</font>] In `config_fixed.py#get_full_config`, I believed the original hyperparameters were too weak for full-scale training. I updated them and logged the changes below the function definition.
7. \[<font color='Red'>To be Solved</font>] In `train_diffpure.py#main`, there is **no mechanism to resume from a checkpoint**. Training always starts from scratch.
8. \[<font color='Red'>To be Solved</font>] In `train_diffpure.py#main`, model is saved only when validation loss improves — but **there is no final save at the end** of training. If the loss fluctuates, final model may be lost.
9. \[<font color='Green'>Solved</font>] In `train_diffpure.py#get_config_for_dataset`, configuration uses `get_debug_config()` as a base, which may unintentionally override externally defined values.
10. \[<font color='Red'>To be Solved</font>] In `train_diffpure.py#main`, `UNet` is always initialized with `hidden_channels=64`. This value should be configurable, ideally via CLI or config.
11. \[<font color='Red'>To be Solved</font>] In `train_diffpure.py#main`, no random seed is set (`random`, `torch`, or `numpy`) — results will be non-deterministic across runs.
12. \[<font color='Red'>To be Solved</font>] In `train_diffpure.py#main`, there is no support for loading a pretrained model — which makes this unsuitable for fine-tuning workflows.
13. \[<font color='Red'>To be Solved</font>] In `train_combined_defense.py#main`, there is no random seed being set — training will be non-reproducible.
14. \[<font color='Red'>To be Solved</font>] In `train_combined_defense.py#main`, there is no exception handling or logging for failures in `run_federated(cfg)` — a failed training would crash silently.
15. \[<font color='Red'>To be Solved</font>] `setup_system.py#check_models_ready` creates and tests the model, but does **not clean up memory afterward**, which could be problematic on low-resource GPUs.
16. \[<font color='Red'>To be Solved</font>] `setup_system.py#check_attacks_ready` falls back silently to `.generate()` if `.forward()` fails. This may mask bugs in the attack API.
17. \[<font color='Yellow'>Partially Solved</font>] `mae_detector1.py#forward` loops over each sample in batch (`for b in range(B)`) for encoder/decoder input prep — this is inefficient and should ideally be vectorized.
18. \[<font color='Red'>To be Solved</font>] `mae_detector1.py#train` computes loss using per-sample masking manually — but **mask indices are not guaranteed to match patch positions** after padding in `x_vis_list`. Risk of incorrect gradients.
19. \[<font color='Red'>To be Solved</font>] `mae_detector1.py#_try_load` does not validate state dict keys (e.g., missing or mismatched shapes could silently fail or produce incorrect weights).
20. \[<font color='Yellow'>Partially Solved</font>] `mae_detector1.py#reconstruction_error` assumes perfect alignment between prediction and ground truth, but **ignores mask usage**. This could misrepresent errors on partially reconstructed inputs.
21. \[<font color='Red'>To be Solved</font>] `mae_detector1.py#MAEEncoder` uses TransformerEncoder directly, but **does not include any normalization layer** (e.g., LayerNorm), which may affect stability.
22. \[<font color='Red'>To be Solved</font>] `mae_detector1.py#MAEDetector.detect` does not clamp or preprocess inputs, leaving vulnerability to out-of-bound tensors from adversarial pipelines.
23. \[<font color='Green'>Solved</font>] `br35h.py#get_br35h_transforms` some commented lines, in train transform. (we don't need them unless looking for extensive augmentation)
24. \[<font color='Green'>Solved</font>] `br35h.py#get_br35h_info` what are those hard-coded means and stds? (they are integrated with Mohammad Nemati)
25.  \[<font color='Purple'>Critical</font>] `train_diffpure.py#get_config_for_dataset`, for a complete training session, we must change the config from debug_config to full config.
26.  \[<font color='Yellow'>Partially Solved</font>] `diffusion/diffuser.py#UNet` The structure of Unet seems valid and well aligned with the dataset we have, but a few things still remain: 1- Maybe adding an additional down and up layers help, or changing the number of hidden channels, which are expected to be examined using empirical testings 2- It may wort adding a fine tuning mechanism to fine tune a bigger diffusion on the current data instead of training a new one from scratch
27.  \[<font color='Green'>Solved</font>] `train_diffpure.py#train_epoch & evaluate` two things changed, first the loss of MSE is moved outside of the loop for better memory utilization, second and more important is, the way the noise is being added to the data is changed and now with each timestep, the noise is being scaled corresponding to that so it helps better learning.
28.  \[<font color='Yellow'>Partially Solved</font>] `train_diffpure.py` The training process for diffusion model can be customized for medical images, things like linear scheduling, some medical and anatomical constraints and changing the distro of noise from gaussian to rician, I won't change them but they can be used to improve the model, just in case.
29.  \[<font color='Red'>To be Solved</font>] `train_diffpure.py` One Important Idea that worth considering, is that we take an already huge diffusion model and only fine tune it on out dataset, this way it takes lot less time and presumably yields us with a better result, worth considering in case of poor result from current model.
30.  \[<font color='Red'>To be Solved</font>] `config_fixed.py` I've found several problems with full and debug configs being generated in this file, though I kept it for the time our system works and gives us a result, then we should return to this and fix the final training config as our last step.
---

## <font color='MediumOrchid'>Coding and Documentation ***Usage*** Issues</font>

### <font color='Thistle'>Coding Usage Issues</font>

1. \[<font color='Red'>To be Resolved</font>] The following modules appear unused. Consider removing them if they're unnecessary:

   * `train_combined_defense.py`, `federated/trainer.py`: These seem to duplicate functionality found in `main.py#run_federated_training`. However, I’m not sure if `run_federated_training` is as complete or correct as these modules. Needs validation.
2. \[<font color='Red'>To be Resolved</font>] Some modules were not included in the final project:

   * `test_config_integration.py`, `simple_test.py`, `server1.py`, `train_combined_defenses.py`, `run_training.py`, `
3. \[<font color='Red'>To be Solved</font>] In `train_diffpure.py#main`, the call to `get_dataset(cfg)` assumes train/test splits exist and work for all datasets — may break with custom datasets (e.g., `br35h`).
4. \[<font color='Red'>To be Solved</font>] In `train_diffpure.py#main`, the training configuration (`args` and `cfg`) is not saved to disk, making it difficult to reproduce experiments.
5. \[<font color='Red'>To be Solved</font>] In `train_combined_defense.py#main`, the script assumes `run_federated(cfg)` is stable and complete — but **there is no sanity check** on whether all required fields (e.g., `cfg.DIFFUSER_STEPS`) are set properly before calling it.
6. \[<font color='Red'>to be Solved</font>] `setup_system.py#check_federated_ready` assumes that `Client` and `Server` creation alone implies readiness — but does not test actual federated updates (e.g., aggregation step).
7. \[<font color='Red'>to be Solved</font>] `mae_detector1.py` is being used very poorly, it is outside of each folder and yet being referenced in `defense/mae_detector.py` and used, this is very confusing.
8. \[<font color='Red'>To be Solved</font>] `mae_detector1.py#MAEDetector.train` No support for **saving best-performing model** (based on val loss or detection accuracy).
9. \[<font color='Red'>To be Solved</font>] `mae_detector1.py#MAEDetector.train` has no batch size compatibility check — potential for CUDA OOM or invalid gradient shape in edge cases.
10. \[<font color='Yellow'>Partially Solved</font>] `mae_detector1.py#reconstruct` is exposed but never used in training/validation — unclear if the reconstruction quality is visually evaluated anywhere.





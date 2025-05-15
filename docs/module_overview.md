# Module Overview

This document provides a brief overview of each top-level directory in the project.

## Top-Level Directories

### `defense/`
Contains implementations of defense mechanisms against adversarial attacks. The primary component is `combined_defense.py` which integrates DiffPure (diffusion-based purification) and pFedDef (personalized federated learning defense) to create a robust defense against adversarial examples.

### `models/`
Contains neural network model definitions. Includes `pfeddef_model.py` which implements the personalized federated defense model with ensemble learners, and `resnet.py` which provides ResNet18 architecture used for classification tasks.

### `utils/`
Houses utility functions and helper classes for the project. This includes data handling utilities in `data_utils.py` and model management utilities in `model_manager.py` that help with loading, saving, and managing model checkpoints.

### `federated/`
Implements federated learning infrastructure including client and server components. The `client.py` handles local model training, `server.py` coordinates model aggregation, and `trainer.py` manages the overall federated learning process.

### `scripts/`
Contains standalone scripts for testing, evaluation, and analysis. The main script is `sanity_suite.py` which runs a series of tests to validate the combined defense performance against adversarial attacks.

### `attacks/`
Houses implementations of adversarial attack algorithms. Includes `pgd.py` (Projected Gradient Descent) and `fgsm.py` (Fast Gradient Sign Method) which are commonly used to generate adversarial examples, as well as `internal_pgd.py` which implements a customized PGD attack.

### `diffusion/`
Implements diffusion models used for image purification in the DiffPure part of the defense. The main component is `diffuser.py` which contains a UNet-based model for noise prediction and a purification process to remove adversarial perturbations.

### `metrics/`
Provides metrics calculation and logging utilities. The `logger.py` module handles tracking, visualizing, and storing various performance metrics during training and evaluation.

### `tests/`
Contains test scripts to verify correct functionality of the codebase. Includes the newly added `quick_imports.py` which checks that all modules can be imported without errors.

## Core Files

### `main.py`
The main entry point for the application. It sets up the federated learning process, initializes clients and server, and orchestrates the training and evaluation of the combined defense.

### `config.py`
Defines the configuration settings and parameters for the project. Contains preset configurations for different scenarios (debug, full) and handles command-line argument parsing.

### `quick_test.py`
A simplified test script for quickly evaluating the combined defense performance. It creates minimal model instances and tests their effectiveness against adversarial examples.

### `train_combined_defense.py`
Script for training the combined defense (DiffPure + pFedDef) with various configuration options. It allows for different presets and has a demo mode for quick testing.

## Project Structure

The project follows a modular architecture, separating concerns into distinct components:

1. **Data handling**: Loading, preprocessing, and batching data
2. **Model definitions**: Neural network architectures and ensemble models
3. **Federated infrastructure**: Client-server communication and aggregation
4. **Defense mechanisms**: Techniques to make models more robust against attacks
5. **Attack algorithms**: Methods to generate adversarial examples
6. **Training and evaluation**: Scripts to train models and evaluate their performance 
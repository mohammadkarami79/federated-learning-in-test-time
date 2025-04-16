# pFedDef: A Guide for Files

## Preparing the Dataset:
- ### CIFAR10:
    - Use the following command to download and preprocess the data. This is the default script, and is subject to change with different settings as we may need. 
    ```powershell
    python ./data/cifar10/generate_data.py  --n_tasks 80 --n_components 3 --alpha 0.4 --s_frac 1.0 --tr_frac 0.8 --seed 12345
    ```
    - The arguments and their meaning can be read in ```data/cifar10/README.md```

## Adversarial Training Run:
### Experiment Collection
- There are several different adversarial training and aggregation methods inside the `run_experiments_collection` directory.
### ```run_experiment_pFedDef.py```
- The file uses ```train_model_input.txt``` for configurations, such as training type, number of clients and many other hyperparameters.
- Default Method: `FedEM_adv` which is equal to `pFedDef`. 

### Components of the Code:
#### 1. Loading and Parsing Input Parameters
   - Reads parameters from `train_model_input.txt` and extracts them using the `revised_input()` function.
   - These parameters define:
     - **Experiment settings:** Dataset name, training method, and number of clients.
     - **Training hyperparameters:** Learning rate, batch size, number of rounds, optimizer, etc.
     - **Adversarial attack settings:** Attack step size (`step_size`), attack strength (`eps`), and how frequently to update adversarial data (`Q`).

#### 2. Client and Aggregator Initialization
   - Calls `dummy_aggregator(args_, num_clients)` to create:
     - **Clients:** These are FL clients that train locally.
     - **Aggregator:** A central server that collects and aggregates model updates.

#### 3. Setting Up Adversarial Attacks (if enabled
   - Checks if the training method is adversarial (`FedAvg_adv`, `FedEM_adv`, `local_adv`).
   - Retrieves `x_min` and `x_max` from a client’s dataset.
   - Defines PGD attack parameters using `PGD_Params()`, which are stored for later use.

#### 4. Computing Data Distribution
   - `Du[i]` stores the number of data points in each client’s dataset.
   - `D = np.sum(Du)` gets the total number of data points across all clients.

#### 5. Training Loop
   - Uses a `while` loop to iterate over `args_.n_rounds`.
   - Calls `aggregator.mix()`, which handles:
     - **Client-side local training**: Calling `client.step()` which trains the model.
     - **Model aggregation at the central server**: With Krum or simple averaging.

   - Every `Q` rounds, the adversarial dataset is updated:
     - Computes hypothesis weights `Whu` for each client.
     - Uses `solve_proportions()` to calculate the adversarial data ratio for each client.
     - Updates each client’s adversarial dataset using `clients[i].assign_advdataset()`.
   - The `PGD_Params()` are set up, and the actual attack happens inside `clients[i].update_advnn()` and `clients[i].assign_advdataset()`.
#### 6. Saving Model State
   - If `args_.save_path` exists, saves model state to disk.
   - Clears memory using `torch.cuda.empty_cache()`.


## Evaluation
### Evaluation Collection
   - Some different evaluation are performed in jupyter notebook instances found in the `Evaluation` directory. 
   - Note that the jupyter notebook environment and package dependency is equivalent to the .py files used to run the experiments.
   - `ens_eval.py` contains the code for Ensemble attack.

   - **Transfer attack** between clients and recording statistic
   - **Ensemble attack**, where multiple clients jointly perform attacks by sharing gradient information as seen in Ensemble adversarial black-box attacks against deep learning systems
   - **Inter-boundary distance measurements** between different models.
   - **Measuring emprical transfer ability** metrics such as gradient alignment.

### ```Ensemble Attack.ipynb```
#### 1. Model Configuration
   - The experiment deploys multiple learners to enhance federated learning performance.
   - Dataset, local steps, the FL setting and other hyperparameters are set.

#### 2. Data Aggregation & Processing
   - A dummy aggregator is created with 40 clients, and the validation data from all clients is combined to form the evaluation dataset. The dataset is then loaded into a custom dataloader for further testing.

#### 3. Model Weight Generation
   - The pre-trained global ensemble models are loaded from the checkpoint directory. 
   - Each learner's model weights are extracted, and different linear combinations of these weights are generated to form test models.
   - Each model is constructed by interpolating the weights from the three learners.

#### 4. Performing Attack
- PGD attack is employed to assess the model’s vulnerability. The attack parameters are set in the code.
- For each of the test models, adversarial examples are generated using PGD, and their effect on victim models (the original ensemble learners) is evaluated.
- Both targeted and untargeted attacks are done.
- Metrics are logged.

## Other Notes
1. Changing models are easy, as you only need to change the loaded model in `get_learner()` function in `utils/utils.py`.
2. Multiple model loaders, such as MobileNet and Resnet, have been implemented, as seen in `models.py`. 
3. PGD and IFGSM are implemented in `transfer_attacks/Personalized_NN.py`.
4. Generating of adversarial data, using them for attacks (pgd) and updating the dataset of each client is implemented in `client.py`. 
5. In FedEM, each client maintains and trains 3 models (learners) instead of a single one. The process works like this:
    - **Local Training**: Each client trains 3 separate models on its own local data. These models (learners) are updated independently but are influenced by the client's data distribution.
    - **Aggregation**: After training locally, clients send their model updates to the server. The server then aggregates the updates across clients while maintaining the mixture of learners.
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from client import Client, MixtureClient
from diffusion_models import DiffusionModel, ScoreNetwork

class DiffPureClient(Client):
    """
    Client that uses Diffusion Purification (DiffPure) to defend against adversarial examples.
    
    This client extends the base Client class by adding a diffusion model for purifying 
    adversarial examples before feeding them to the classifier.
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            diffusion_model=None,
            t_star=0.1,
            n_steps=100,
            tune_locally=False,
            tune_steps=None
    ):
        """
        Initialize DiffPure client.
        
        Args:
            learners_ensemble: Ensemble of learner models
            train_iterator: Data iterator for training data
            val_iterator: Data iterator for validation data
            test_iterator: Data iterator for test data
            logger: Logger for tracking metrics
            local_steps: Number of local steps per round
            diffusion_model: Pre-trained diffusion model, if None will create a new one
            t_star: Diffusion timestep for purification
            n_steps: Number of reverse diffusion steps
            tune_locally: Whether to tune models locally
            tune_steps: Number of tuning steps
        """
        super().__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )
        
        # Get device from learners_ensemble
        self.device = next(self.learners_ensemble.learners[0].model.parameters()).device
        
        # Set up diffusion model
        if diffusion_model is None:
            # Create score network
            in_channels = 3  # Assuming RGB images
            score_network = ScoreNetwork(in_channels=in_channels).to(self.device)
            
            # Create diffusion model
            self.diffusion_model = DiffusionModel(score_network, device=self.device)
            
            # Flag indicating whether the diffusion model is trained
            self.diffusion_trained = False
        else:
            self.diffusion_model = diffusion_model
            self.diffusion_trained = True
            
        self.t_star = t_star
        self.n_steps = n_steps
    
    def train_diffusion_model(self, epochs=10, lr=1e-4):
        """
        Train the diffusion model using the client's training data.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Trained diffusion model
        """
        from diffusion_models import train_diffusion_model
        
        self.diffusion_model.model = train_diffusion_model(
            self.diffusion_model.model,
            self.train_iterator, 
            epochs=epochs,
            lr=lr,
            device=self.device
        )
        
        self.diffusion_trained = True
        return self.diffusion_model
    
    def purify_batch(self, batch):
        """
        Purify a batch of data using the diffusion model.
        
        Args:
            batch: Batch of data (inputs, targets)
            
        Returns:
            Purified batch
        """
        if not self.diffusion_trained:
            return batch
        
        inputs, targets = batch
        inputs = inputs.to(self.device)
        
        # Purify inputs using diffusion model
        purified_inputs = self.diffusion_model.purify(
            inputs, 
            t_star=self.t_star,
            n_steps=self.n_steps
        )
        
        return purified_inputs, targets
    
    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        Perform a step for the client, with DiffPure purification.
        
        Args:
            single_batch_flag: If true, only use one batch
            
        Returns:
            Client updates
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()

        if single_batch_flag:
            batch = self.get_next_batch()
            
            # Purify batch if diffusion model is trained
            if self.diffusion_trained:
                batch = self.purify_batch(batch)
                
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            # Create a wrapper iterator that purifies batches on-the-fly
            if self.diffusion_trained:
                original_iterator = self.train_iterator
                
                # Create a purified iterator
                class PurifiedIterator:
                    def __init__(self, original_iterator, diffpure_client):
                        self.original_iterator = original_iterator
                        self.diffpure_client = diffpure_client
                        self.dataset = original_iterator.dataset
                        
                    def __iter__(self):
                        for batch in self.original_iterator:
                            yield self.diffpure_client.purify_batch(batch)
                            
                    def __len__(self):
                        return len(self.original_iterator)
                
                self.train_iterator = PurifiedIterator(original_iterator, self)
            
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights
                )
            
            # Restore original iterator
            if self.diffusion_trained:
                self.train_iterator = original_iterator

        return client_updates
    
    def evaluate(self, iterator):
        """
        Evaluate the model on the iterator with purification.
        
        Args:
            iterator: Data iterator
            
        Returns:
            loss, accuracy
        """
        if not self.diffusion_trained:
            return self.learners_ensemble.evaluate_iterator(iterator)
        
        # Create a purified version of the iterator
        class PurifiedEvalIterator:
            def __init__(self, original_iterator, diffpure_client):
                self.original_iterator = original_iterator
                self.diffpure_client = diffpure_client
                self.dataset = original_iterator.dataset
                
            def __iter__(self):
                for batch in self.original_iterator:
                    yield self.diffpure_client.purify_batch(batch)
                    
            def __len__(self):
                return len(self.original_iterator)
        
        purified_iterator = PurifiedEvalIterator(iterator, self)
        
        return self.learners_ensemble.evaluate_iterator(purified_iterator)
    
    def write_logs(self):
        """
        Write logs for the client, with purified evaluation.
        """
        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.evaluate(self.val_iterator)
            test_loss, test_acc = self.evaluate(self.test_iterator)
        else:
            train_loss, train_acc = self.evaluate(self.val_iterator)
            test_loss, test_acc = self.evaluate(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc


class DiffPureMixtureClient(DiffPureClient, MixtureClient):
    """
    Client that combines MixtureClient functionality with DiffPure purification.
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            diffusion_model=None,
            t_star=0.1,
            n_steps=100,
            tune_locally=False,
            tune_steps=None
    ):
        DiffPureClient.__init__(
            self,
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            diffusion_model=diffusion_model,
            t_star=t_star,
            n_steps=n_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )
    
    def update_sample_weights(self):
        """
        Update sample weights using MixtureClient's method.
        """
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T 
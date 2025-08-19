import contextlib
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Union
from abc import abstractmethod
from tqdm import tqdm
from copy import deepcopy

from dgn4avbp.diffusion_process import DiffusionProcess, DiffusionProcessSubSet
from dgn4avbp.step_sampler import ImportanceStepSampler
#from ...graph import Data
from dgn4avbp.loader import DataLoader, Collater

from dgn4avbp.blocks import SinusoidalPositionEmbedding
from dgn4avbp.multi_scale_gnn import MultiScaleGnn
from torch_geometric.data import Data



class Model(nn.Module):
    r"""Base class for all the GNN models.

    Args:
        arch (Optional[Union[None,dict]], optional): Dictionary with the model architecture. Defaults to `None`.
        weights (Optional[Union[None,str]], optional): Path of the weights file. Defaults to `None`.
        model (Optional[Union[None,str]], optional): Path of the checkpoint file. Defaults to `None`.
        device (Optional[torch.device], optional): Device where the model is loaded. Defaults to `torch.device('cpu')`.
    """

    def __init__(
        self,
        arch:       dict         = None,
        weights:    str          = None,
        checkpoint: str          = None,
        device:     torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__()
        self.device = device
        self.load_model(arch, weights, checkpoint)

    # To be overwritten
    @property
    def num_fields(self) -> int:
        """Returns the number of output fields. It must be overloaded by each model instancing the `graphs4cfd.models.Model` class."""
        pass

    def load_model(self, arch, weights, checkpoint):
        """Loads the model architecture from a arch dictionary and its weights from a weights file, or loads the model from a checkpoint file."""
        if arch is not None and checkpoint is None:
            self.load_arch(arch)
            # To device
            self.to(self.device)
            if weights is not None:
                self.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
      
        elif arch is None and weights is None and checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if 'diffusion_process' in checkpoint.keys():
                self.diffusion_process = checkpoint['diffusion_process']
            if 'learnable_variance' in checkpoint.keys():
                self.learnable_variance = checkpoint['learnable_variance']
            self.load_arch(checkpoint['arch'])
            self.to(self.device)
            self.load_state_dict(checkpoint['weights'])
        return

    # To be overwritten
    def load_arch(self, arch: dict):
        """Defines the hyper-parameters of the model. It must be overloaded by each model instancing the `graphs4cfd.models.Model` class.

        Args:
            arch (dict): Dictionary with the architecture of the model. Its structure depends on the model.
        """
        pass
    
    # To be overwritten
    def forward(self, graph: Data):
        """Forwrad pass (or time step) of the model. It must be overloaded by each model instancating the `graphs4cfd.models.Model` class.

        Args:
            graph (Data): Data object with the input data.
            t (int): current time-point
        """
        pass
 
    def fit(
        self,
        trainining_settings: object,
        train_loader:        DataLoader,
        val_loader:          DataLoader = None
    ) -> None:
        """Trains the model.
        
        Args:
            config       (TrainingSettings):               Configuration of the training.
            train_loader (DataLoader):                     Training data loader.
            val_loader   (Optional[DataLoader], optional): Validation data loader. Defaults to `None`.
        """
        # Change the training device if needed
        if trainining_settings['device'] is not None and trainining_settings['device'] != self.device:
            self.to(trainining_settings['device'])
            self.device = trainining_settings['device']
        # Set the training loss
        criterion = trainining_settings['training_loss']
        # Load checkpoint
        checkpoint = None
        scheduler  = None
        if trainining_settings['checkpoint'] is not None and os.path.exists(trainining_settings['checkpoint']):
            print("Training from an existing checkpoint:", trainining_settings['checkpoint'])
            checkpoint = torch.load(trainining_settings['checkpoint'], map_location=self.device, weights_only=True)
            self.load_state_dict(checkpoint['weights'])
            optimiser = torch.optim.Adam(self.parameters(), lr=checkpoint['lr'])
            optimiser.load_state_dict(checkpoint['optimiser'])
            if trainining_settings['scheduler'] is not None: 
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=trainining_settings['scheduler']['factor'], patience=trainining_settings['scheduler']['patience'], eps=0.)
                scheduler.load_state_dict(checkpoint['scheduler'])
            initial_epoch = checkpoint['epoch'] + 1
        # Initialise optimiser and scheduler if not previous checkpoint is used
        else:
            # If a .chk is given but it does not exist such file, notify the user
            if trainining_settings['checkpoint'] is not None:
                print("Not matching checkpoint file:", trainining_settings['checkpoint'])
            print('Training from randomly initialised weights')
            optimiser = optim.Adam(self.parameters(), lr=trainining_settings['lr'])
            if trainining_settings['scheduler'] is not None: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=trainining_settings['scheduler']['factor'], patience=trainining_settings['scheduler']['patience'], eps=0.)
            initial_epoch = 1
        # If .chk to save exists rename the old version to .bck
        path = os.path.join(trainining_settings["folder"], trainining_settings["name"]+".chk")
        if os.path.exists(path):
            print('Renaming', path, 'to:', path+'.bck')
            os.rename(path, path+'.bck')
        # Initialise tensor board writer
        if trainining_settings['tensor_board'] is not None: writer = SummaryWriter(os.path.join(trainining_settings["tensor_board"], trainining_settings["name"]))
        # Initialise automatic mixed-precision training
        scaler = None
        if trainining_settings['mixed_precision']:
            print("Training with automatic mixed-precision")
            scaler = torch.cuda.amp.GradScaler()
            # Load previos scaler
            if checkpoint is not None and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
        # Print before training
        print(f'Training on device: {self.device}')
        print(f'Number of learnable parameters: {self.num_learnable_params}')
        print(f'Total number of parameters:     {self.num_params}')
        # Training loop
        for epoch in tqdm(range(initial_epoch,trainining_settings['epochs']+1), desc="Completed epochs", leave=False, position=0):
            if optimiser.param_groups[0]['lr'] < trainining_settings['stopping']:
                print(f"The learning rate is smaller than {trainining_settings['stopping']}. Stopping training.")
                self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
                break
            print("\n")
            print(f"Hyperparameters: lr = {optimiser.param_groups[0]['lr']}")
            self.train()
            training_loss = 0.
            gradients_norm = 0.
            for iteration, data in enumerate(train_loader):
                data = data.to(self.device)
                with torch.cuda.amp.autocast() if trainining_settings['mixed_precision'] else contextlib.nullcontext(): # Use automatic mixed-precision
                    loss = criterion(self, data)
                # Back-propagation
                if trainining_settings['mixed_precision']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                # Save training loss and gradients norm before applying gradient clipping to the weights
                training_loss  += loss.item()
                gradients_norm += self.grad_norm()
                # Update the weights
                if trainining_settings['mixed_precision']:
                    # Clip the gradients
                    if trainining_settings['grad_clip'] is not None and epoch > trainining_settings['grad_clip']["epoch"]:
                        scaler.unscale_(optimiser)
                        nn.utils.clip_grad_norm_(self.parameters(), trainining_settings['grad_clip']["limit"])
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    # Clip the gradients
                    if trainining_settings['grad_clip'] is not None and epoch > trainining_settings['grad_clip']["epoch"]:
                        nn.utils.clip_grad_norm_(self.parameters(), trainining_settings['grad_clip']["limit"])
                    optimiser.step()
                # Reset the gradients
                optimiser.zero_grad()
            training_loss  /= (iteration + 1)
            gradients_norm /= (iteration + 1)
            # Display on terminal
            print(f"Epoch: {epoch:4d}, Training   loss: {training_loss:.4e}, Gradients: {gradients_norm:.4e}")
            # Testing
            if val_loader is not None:
                validation_criterion = trainining_settings['validation_loss']
                self.eval()
                with torch.no_grad(): 
                    validation_loss = 0.
                    for iteration, data in enumerate(val_loader):
                        data = data.to(self.device)
                        validation_loss += validation_criterion(self, data).item()
                    validation_loss /= (iteration+1)
                    print(f"Epoch: {epoch:4d}, Validation loss: {validation_loss:.4e}")
            # Log in TensorBoard
            if trainining_settings['tensor_board'] is not None:
                writer.add_scalar('Loss/train', training_loss,   epoch)
                if val_loader: writer.add_scalar('Loss/test',  validation_loss, epoch)
            # Update lr
            if trainining_settings['scheduler']['loss'][:2] == 'tr':
                scheduler_loss = training_loss 
            elif trainining_settings['scheduler']['loss'][:3] == 'val':
                scheduler_loss = validation_loss 
            scheduler.step(scheduler_loss)
            # Create training checkpoint
            if not epoch % trainining_settings["chk_interval"]:
                print('Saving checkpoint in:', path)
                self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
        # Save final checkpoint
        self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
        writer.close()
        print("Finished training")
        return

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        optimiser: torch.optim.Optimizer,
        scheduler: Union[None, dict] = None,
        scaler: Union[None, dict] = None
    ) -> None:
        """Saves the model parameters, the optimiser state and the current value of the training hyper-parameters.
        The saved file can be used to resume training with the `graphs4cfd.nn.model.Model.fit` method."""
        checkpoint = {
            'weights'  : self.state_dict(),
            'optimiser': optimiser.state_dict(),
            'lr'       : optimiser.param_groups[0]['lr'],
            'epoch'    : epoch,
        }
        if hasattr(self, 'arch'):               checkpoint['arch']               = self.arch
        if hasattr(self, 'diffusion_process'):  checkpoint['diffusion_process']  = self.diffusion_process
        if hasattr(self, 'learnable_variance'): checkpoint['learnable_variance'] = self.learnable_variance
        if scheduler is not None: checkpoint['scheduler'] = scheduler.state_dict()
        if scaler    is not None: checkpoint['scaler']    = scaler.state_dict()
        torch.save(checkpoint, filename)
        return

    @property
    def num_params(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_learnable_params(self):
        """Returns the number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def grad_norm(self):
        """Returns the L2 norm of the gradients."""
        norm = 0.
        for p in self.parameters():
            if p.requires_grad:
                norm += p.grad.data.norm(2).item()**2
        return norm**.5



class DiffusionModel(Model):
    r"""Abstract class for diffusion models. This class implements the training loop and the sampling method. The forward pass must be implemented in the derived class.

    Args:
        diffusion_process (DiffusionProcess, optional): The diffusion process. It must be provided if it we are not loading a checkpoint. Defaults to `None`.
        learnable_variance (bool, optional): If `True`, the variance is learnable. If we are loading a checkpoint, this value must be the same as the one in the checkpoint. 
            If `False`, the model output the noise, as in "Denoising Diffusion Probabilistic Models". If `True`, the model output the noise and the variance, as in "Improved Denoising Diffusion Probabilistic Models".
            Defaults to `False`.

    Methods:
        fit: Train the model using the provided training settings and data loader.
        forward: Forward pass of the model. This method must be implemented for each model.
        get_posterior_mean_and_variance_from_output: Compute the posterior mean and variance from the model output.
        sample: Sample from the model.
    """


    def __init__(
        self,
        diffusion_process:  DiffusionProcess = None,
        learnable_variance: bool             = None,
        *args, 
        **kwargs
    ) -> None:
        self.learnable_variance = learnable_variance
        super().__init__(*args, **kwargs)
        # This may overwrite the diffusion_process and learnable_variance if we are loading a checkpoint.
        # So, we need to check if the provided values are the same as the ones in the model.
        if diffusion_process is None and not hasattr(self, 'diffusion_process'):
            raise RuntimeError('diffusion_process must be provided')
        elif diffusion_process is not None:
            if hasattr(self, 'diffusion_process'):
                print('Warning: diffusion_process is provided, but it is already defined in the model.')
                print('The provided diffusion_process will be used.')
            self.diffusion_process = diffusion_process
        if learnable_variance is None and not hasattr(self, 'learnable_variance'):
            raise RuntimeError('learnable_variance must be provided')
        if learnable_variance is not None and learnable_variance != self.learnable_variance:
            raise RuntimeError('learnable_variance is different from the one in the provided checkpoint')
  
    @property
    def is_latent_diffusion(self):
        return hasattr(self, 'autoencoder')

    def fit(
        self,
        training_settings: object,
        dataloader:        DataLoader,
    ) -> None:
        """Train the model using the provided training settings and data loader.

        Args:
            training_settings (TrainingSettings): The training settings.
            dataloader (DataLoader): The data loader.
        """
        # Verify the training settings
        if training_settings['scheduler']['loss'][:3].lower() == 'val':
            raise NotImplementedError("Wrong training settings: Validation loss is not implemented yet.")
        # Change the training device if needed
        if training_settings['device'] is not None and training_settings['device'] != self.device:
            self.to(training_settings['device'])
            self.device = training_settings['device']
        # Set the diffusion step sampler
        step_sampler = training_settings['step_sampler'](num_diffusion_steps = self.diffusion_process.num_steps)
        # Set the training loss
        criterion = training_settings['training_loss']
        # Load checkpoint
        checkpoint = None
        scheduler  = None
        if training_settings['checkpoint'] is not None and os.path.exists(training_settings['checkpoint']):
            print("Training from an existing check-point:", training_settings['checkpoint'])
            checkpoint = torch.load(training_settings['checkpoint'], map_location=self.device)
            self.load_state_dict(checkpoint['weights'])
            optimiser = torch.optim.Adam(self.parameters(), lr=checkpoint['lr'])
            optimiser.load_state_dict(checkpoint['optimiser'])
            if training_settings['scheduler'] is not None: 
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=training_settings['scheduler']['factor'], patience=training_settings['scheduler']['patience'], eps=0.)
                scheduler.load_state_dict(checkpoint['scheduler'])
            initial_epoch = checkpoint['epoch'] + 1
        # Initialise optimiser and scheduler if not previous check-point is used
        else:
            # If a .chk is given but it does not exist such file, notify the user
            if training_settings['checkpoint'] is not None:
                print("Not matching check-point file:", training_settings['checkpoint'])
            print('Training from randomly initialised weights.')
            optimiser = optim.Adam(self.parameters(), lr=training_settings['lr'])
            if training_settings['scheduler'] is not None: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=training_settings['scheduler']['factor'], patience=training_settings['scheduler']['patience'], eps=0.)
            initial_epoch = 1
        # If .chk to save exists rename the old version to .bck
        path = os.path.join(training_settings["folder"], training_settings["name"]+".chk")
        if os.path.exists(path):
            print('Renaming', path, 'to:', path+'.bck')
            os.rename(path, path+'.bck')
        # Initialise tensor board writer
        if training_settings['tensor_board'] is not None: writer = SummaryWriter(os.path.join(training_settings["tensor_board"], training_settings["name"]))
        # Initialise automatic mixed-precision training
        scaler = None
        if training_settings['mixed_precision']:
            print("Training with automatic mixed-precision")
            scaler = torch.cuda.amp.GradScaler()
            # Load previos scaler
            if checkpoint is not None and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
        # Print before training
        print(f'Training on device: {self.device}')
        print(f'Number of learnable parameters: {self.num_learnable_params}')
        print(f'Total number of parameters:     {self.num_params}')
        # Training loop
        for epoch in tqdm(range(initial_epoch, training_settings['epochs']+1), desc="Completed epochs", leave=False, position=0):
            if optimiser.param_groups[0]['lr'] < training_settings['stopping']:
                print(f"The learning rate is smaller than {training_settings['stopping']}. Stopping training.")
                self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
                break
            print("\n")
            print(f"Hyperparameters: lr = {optimiser.param_groups[0]['lr']}")
            self.train()
            training_loss = 0.
            gradients_norm = 0.
            for iteration, graph in enumerate(dataloader):
                graph = graph.to(self.device)
                batch_size = graph.batch.max().item() + 1
                if self.is_latent_diffusion:
                    graph = self.autoencoder.transform(graph)
                # Forward pass
                with torch.cuda.amp.autocast() if training_settings['mixed_precision'] else contextlib.nullcontext(): # Use automatic mixed-precision
                    # Sample a batch of random diffusion steps
                    graph.r, sample_weight = step_sampler(batch_size, self.device) # Dimension: (batch_size), (batch_size)
                    # Diffuse the solution/target field
                    graph.field_start = graph.x_latent_target if self.is_latent_diffusion else graph.target
                    graph.field_r, graph.noise = self.diffusion_process(
                        field_start    = graph.field_start,
                        r              = graph.r,
                        batch          = graph.batch,
                        dirichlet_mask = None if self.is_latent_diffusion else getattr(graph, 'dirichlet_mask', None)
                    ) # Shapes (num_nodes, num_fields), (num_nodes, num_fields)
                    # Compute the loss for each sample in the batch
                    loss = criterion(self, graph) # Dimension: (batch_size)
                    # Update the loss-aware diffusion-step sampler
                    if isinstance(step_sampler, ImportanceStepSampler):
                        step_sampler.update(graph.r, loss.detach())
                    # Compute the weighted loss over the batch
                    loss = (loss * sample_weight).mean()
                # Back-propagation
                if training_settings['mixed_precision']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                # Save training loss and gradients norm before applying gradient clipping to the weights
                training_loss  += loss.item()
                gradients_norm += self.grad_norm()
                # Update the weights
                if training_settings['mixed_precision']:
                    # Clip the gradients
                    if training_settings['grad_clip'] is not None and epoch > training_settings['grad_clip']["epoch"]:
                        scaler.unscale_(optimiser)
                        nn.utils.clip_grad_norm_(self.parameters(), training_settings['grad_clip']["limit"])
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    # Clip the gradients
                    if training_settings['grad_clip'] is not None and epoch > training_settings['grad_clip']["epoch"]:
                        nn.utils.clip_grad_norm_(self.parameters(), training_settings['grad_clip']["limit"])
                    optimiser.step()
                # Reset the gradients
                optimiser.zero_grad()
            training_loss  /= (iteration + 1)
            gradients_norm /= (iteration + 1)
            # Display on terminal
            print(f"Epoch: {epoch:4d}, Training loss: {training_loss:.4e}, Gradients: {gradients_norm:.4e}")
            # Log in TensorBoard
            if training_settings['tensor_board'] is not None:
                writer.add_scalar('Loss/train', training_loss,   epoch)
            # Update lr
            if scheduler is not None:
                scheduler.step(training_loss)
            # Create training checkpoint
            if not epoch % training_settings["chk_interval"]:
                print('Saving checkpoint in:', path)
                self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
        writer.close()
        print("Finished training")
        return
    
    @abstractmethod
    def forward(self, graph: Data) -> torch.Tensor:
        """Forward pass of the model. This method must be implemented for each model."""
        pass

    def get_posterior_mean_and_variance_from_output(
        self,
        model_output:      Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        graph:             Data,
        diffusion_process: DiffusionProcess = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior mean and variance from the model output.

        Args:
            model_output (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The model output.
                This can be the noise ("Denoising Diffusion Probabilistic Models") or a tuple with the noise and v ("Improved Denoising Diffusion Probabilistic Models")
            graph (Data): The graph.
            diffusion_process (DiffusionProcess, optional): The diffusion process. If `None`, the diffusion process of the model is used. Defaults to `None`.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The posterior mean and variance.
        """
        dp = self.diffusion_process if diffusion_process is None else diffusion_process
        if isinstance(model_output, tuple):
            assert self.learnable_variance, 'The model output is a tuple, but learnable_variance=False'
        else:
            assert not self.learnable_variance, 'The model output is not a tuple, but learnable_variance=True'
        # Pre-compute needed coefficients
        betas_r = dp.get_index_from_list(dp.betas, graph.batch, graph.r)
        sqrt_one_minus_alphas_cumprod_r = dp.get_index_from_list(dp.sqrt_one_minus_alphas_cumprod, graph.batch, graph.r)
        sqrt_recip_alphas_r = dp.get_index_from_list(dp.sqrt_recip_alphas, graph.batch, graph.r)
        if self.learnable_variance:
            eps, v = model_output
            v = (v + 1) / 2
            # For min_log we use the clipped posterior variance to avoid nan values in the backward pass
            min_log = dp.get_index_from_list(dp.posterior_log_variance_clipped, graph.batch, graph.r)
            max_log = torch.log(dp.get_index_from_list(dp.betas, graph.batch, graph.r))
            log_variance = v * max_log + (1 - v) * min_log
            variance = torch.exp(log_variance)
        else:
            eps = model_output
            variance = dp.get_index_from_list(dp.posterior_variance, graph.batch, graph.r)
        mean =  sqrt_recip_alphas_r * (graph.field_r - betas_r * eps/sqrt_one_minus_alphas_cumprod_r)
        return mean, variance

    @torch.no_grad()
    def sample(
        self,
        graph:            Data,
        steps:            list[int]    = None,
        dirichlet_values: torch.Tensor = None,
    ) -> torch.Tensor:
        """Sample from the model.
        
        Args:
            graph (Data): The graph.
            steps (list[int], optional): The steps to sample. If `None`, all steps are sampled. Defaults to `None`.
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary conditions. If `None`, no Dirichlet boundary conditions are applied. Defaults to `None`.

        Returns:
            torch.Tensor: The sample.
        """

        if steps is not None:
            assert all([isinstance(s, int) for s in steps]), 'steps must be a list of integers'
            # Sort the steps in ascending order
            steps = sorted(steps)
            # We cannot have repeated steps and the last step must be smaller than the number of steps in the "full" diffusion process
            assert len(steps) == len(set(steps)), 'steps must not have repeated values'
            assert steps[-1] < self.diffusion_process.num_steps, 'The last step in steps must be smaller than the number of steps in the "full" diffusion process'
        self.eval()
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
        if graph.pos.device != self.device:
            graph.to(self.device)
        if dirichlet_values is not None:
            dirichlet_values = dirichlet_values.to(self.device)
        batch_size = graph.batch.max().item() + 1
        # Get the latent features if the model is a latent diffusion model
        if self.is_latent_diffusion:
            c_latent_list, e_latent_list, edge_index_list, batch_list = self.autoencoder.cond_encoder(
                graph,
                torch.cat([f for f in [graph.get('loc'), graph.get('glob'), graph.get('omega')] if f is not None], dim=1),
                torch.cat([f for f in [graph.get('edge_attr'), graph.get('edge_cond')] if f is not None], dim=1),
                graph.edge_index
            )
            graph.c_latent   = c_latent_list  [-1].clone()
            graph.e_latent   = e_latent_list  [-1].clone()
            graph.edge_index = edge_index_list[-1].clone()
            graph.batch      = batch_list     [-1].clone()
        # Use the spaced diffusion process if it is defined
        dp = DiffusionProcessSubSet(self.diffusion_process, steps) if steps is not None else self.diffusion_process
        # Sample the noisy solution
        graph.field_r = torch.randn(graph.batch.size(0), self.num_fields, device=self.device) # Shape (num_nodes, num_fields)
        # Take into account the dirichlet boundary conditions if they are defined and if we are not using a latent diffusion model
        if hasattr(graph, 'dirichlet_mask') and not self.is_latent_diffusion:
            assert dirichlet_values is not None, 'dirichlet_values must be provided if graph has dirichlet_mask'
            assert dirichlet_values.shape == (graph.num_nodes, self.num_fields), f'dirichlet_values.shape must be (num_nodes, num_fields), but it is {dirichlet_values.shape}'
            dirichlet_field_r, _ = dp(
                field_start    = dirichlet_values,
                r              = torch.tensor(batch_size * [dp.num_steps - 1], dtype=torch.long, device=self.device),
                batch          = graph.batch,
                dirichlet_mask = graph.dirichlet_mask
            ) # Shapes (num_nodes, num_fields), (num_nodes, num_fields)
            graph.field_r = graph.field_r * (~graph.dirichlet_mask) + dirichlet_field_r * graph.dirichlet_mask
        # Denoise 'field_r' step-by-step
        for r in dp.steps[::-1]:
            graph.r = torch.full((batch_size,), r, dtype=torch.long, device=self.device)
            # Compute the posterior mean and variance
            model_output = self(graph)
            mean, variance = self.get_posterior_mean_and_variance_from_output(model_output, graph, dp)
            if hasattr(graph, 'dirichlet_mask') and not self.is_latent_diffusion:
                gaussian_noise = torch.randn_like(mean) *  (~graph.dirichlet_mask) # Shape (num_nodes, num_fields)
            else:
                gaussian_noise = torch.randn_like(mean) # Shape (num_nodes, num_fields)
            graph.field_r = mean + torch.sqrt(variance) * gaussian_noise
        # Decode the denoised latent features
        if self.is_latent_diffusion:
            return self.autoencoder.decode(
                graph           = graph,
                v_latent        = graph.field_r,
                c_latent_list   = c_latent_list,
                e_latent_list   = e_latent_list,
                edge_index_list = edge_index_list,
                batch_list      = batch_list,
                dirichlet_mask  = graph.dirichlet_mask if hasattr(graph, 'dirichlet_mask') else None,
                v_0             = dirichlet_values     if hasattr(graph, 'dirichlet_mask') else None,
            )
        else:
            return graph.field_r
        
    @torch.no_grad()    
    def sample_n(
        self,
        num_samples:      int,
        graph:            Data,
        dirichlet_values: torch.Tensor = None,
        batch_size:       int = 0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Sample `num_samples` samples from the model.

        Args:
            num_samples (int): The number of samples.
            graph (Data): The graph.
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary conditions. If `None`, no Dirichlet boundary conditions are applied. Defaults to `None`.
            batch_size (int, optional): Number of samples to generate in parallel. If `batch_size < 2`, the samples are generated one by one. Defaults to `0`.

        Returns:
            torch.Tensor: The samples. Dimension: (num_nodes, num_samples, num_fields
        """
        samples = []
        # Create (num_samples // num_workers) mini-batches with the same graph repeated num_workers times
        if batch_size > 1:
            collater = Collater()
            num_evals = num_samples // batch_size + (num_samples % batch_size > 0)
            for _ in tqdm(range(num_evals), desc=f"Generating {num_samples} samples", leave=False, position=0):
                current_batch_size = min(batch_size, num_samples - len(samples))
                batch = collater.collate([deepcopy(graph) for _ in range(current_batch_size)])
                # Sample
                sample = self.sample(batch, dirichlet_values=dirichlet_values.repeat(current_batch_size, 1) if dirichlet_values is not None else None, *args, **kwargs)
                # Split base on the batch index
                sample = torch.stack(sample.chunk(current_batch_size, dim=0), dim=1)
                samples.append(sample)
            return torch.cat(samples, dim=1)
        else:
            for _ in tqdm(range(num_samples), desc=f"Generating {num_samples} samples", leave=False, position=0):
                sample = self.sample(graph, dirichlet_values, *args, **kwargs)
                samples.append(sample)
            return torch.stack(samples, dim=1) # Dimension: (num_nodes, num_samples, num_fields)
        

class DiffusionGraphNet(DiffusionModel):
    r"""Defines a GNN that parameterizes the inverse diffusion process.

    Args:
        arch (dict): Dictionary with the architecture of the model. It must contain the following keys:
            - 'in_node_features' (int): Number of input node features. This is the number of features of the noisy field.
            - 'cond_node_features' (int, optional): Number of conditional node features. Defaults to 0.
            - 'cond_edge_features' (int, optional): Number of conditional edge features. Defaults to 0.
            - 'depths' (list): List of integers with the number of layers at each depth.
            - 'fnns_depth' (int, optional): Number of layers in the FNNs. Defaults to 2.
            - 'fnns_width' (int): Width of the FNNs.
            - 'aggr' (str, optional): Aggregation method. Defaults to 'mean'.
            - 'dropout' (float, optional): Dropout probability. Defaults to 0.0.
            - 'emb_width' (int, optional): Width of the diffusion-step embedding. Defaults to 4 * fnns_width.
            - 'dim' (int, optional): Dimension of the latent space. Defaults to 2.
            - 'scalar_rel_pos' (bool, optional): Whether to use scalar relative positions. Defaults to True.
        """

    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Hyperparameters
        self.in_node_features   = arch['in_node_features']
        self.cond_node_features = arch.get('cond_node_features', 0)
        self.cond_edge_features = arch.get('cond_edge_features', 0)
        self.depths             = arch['depths']
        self.fnns_depth         = arch.get('fnns_depth', 2)
        self.fnns_width         = arch['fnns_width']
        self.aggr               = arch.get('aggr', 'mean')
        self.dropout            = arch.get('dropout', 0.0)
        self.emb_width          = arch.get('emb_width', self.fnns_width * 4)
        self.dim                = arch.get('dim', 2)
        self.scalar_rel_pos     = arch.get('scalar_rel_pos', True)
        if 'in_edge_features' in arch: # To support backward compatibility
             self.cond_edge_features = arch['in_edge_features'] + self.cond_edge_features
        # Validate the inputs
        assert self.in_node_features > 0, "Input node features must be a positive integer"
        assert self.cond_node_features >= 0, "Condition features must be a non-negative integer"
        assert len(self.depths) > 0, "Depths (`depths`) must be a list of integers"
        assert isinstance(self.depths, list), "Depths (`depths`) must be a list of integers"
        assert all([isinstance(depth, int) for depth in self.depths]), "Depths (`depths`) must be a list of integers"
        assert all([depth > 0 for depth in self.depths]), "Depths (`depths`) must be a list of positive integers"
        assert self.fnns_depth >=2 , "FNNs depth (`fnns_depth`) must be at least 2"
        assert self.fnns_width > 0, "FNNs width (`fnns_width`) must be a positive integer"
        assert self.aggr in ('mean', 'sum'), "Aggregation method (`aggr`) must be either 'mean' or 'sum'"
        assert self.dropout >= 0.0 and self.dropout < 1.0, "Dropout (`dropout`) must be a float between 0.0 and 1.0"
        self.out_node_features = self.in_node_features * 2 if self.learnable_variance else self.in_node_features
        # Diffusion-step embedding
        self.diffusion_step_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(self.fnns_width),
            nn.Linear(self.fnns_width, self.emb_width),
            nn.SELU(),
        )
        # Node encoder
        self.node_encoder = nn.Linear(
            in_features  = self.in_node_features + self.cond_node_features,
            out_features = self.fnns_width,
        )
        # Diffusion-step encoder
        self.diffusion_step_encoder = nn.ModuleList([
            nn.Linear(self.emb_width, self.fnns_width),      # Applied to the diffusion-step embedding
            nn.SELU(),                                       # Applied to the previous output and the node encoder output
            nn.Linear(self.fnns_width * 2, self.fnns_width), # Applied to the previous output
        ])
        # Edge encoder
        self.edge_encoder = nn.Linear(
            in_features  = self.cond_edge_features,
            out_features = self.fnns_width,
        )
        # MuS-GNN propagator
        self.propagator = MultiScaleGnn(
            depths            = self.depths,
            fnns_depth        = self.fnns_depth,
            fnns_width        = self.fnns_width,
            emb_features      = self.emb_width,
            aggr              = self.aggr,
            activation        = nn.SELU,
            dropout           = self.dropout,
            dim               = self.dim,
            scalar_rel_pos    = self.scalar_rel_pos,
        )
        # Node decoder
        self.node_decoder = nn.Linear(
            in_features  = self.fnns_width,
            out_features = self.out_node_features,
        )
 
    @property
    def num_fields(self) -> int:
        return self.out_node_features // 2 if self.learnable_variance else self.out_node_features
    
    def reset_parameters(self):
            modules = [module for module in self.children() if hasattr(module, 'reset_parameters')]
            for module in modules:
                module.reset_parameters()

    def forward(
        self,
        graph: Data,
    ) -> torch.Tensor:
        assert hasattr(graph, 'r'), "graph must have an attribute 'r' indicating the diffusion step"
        assert hasattr(graph, 'field_r'), "graph must have an attribute 'field_r' corresponding to the field at the previous diffusion step"
        # Embed the diffusion step
        emb = self.diffusion_step_embedding(graph.r) # Shape (batch_size, emb_width)
        # Encode the node features
        v = self.node_encoder(
            torch.cat([
                graph.field_r,
                *[f for f in [graph.get('cond'), graph.get('field'), graph.get('loc'), graph.get('glob'), graph.get('omega')] if f is not None],
            ], dim=1)
        ) # Shape (num_nodes, fnns_width)
        # Encode the diffusion step embedding into the node features
        emb_proj = self.diffusion_step_encoder[0](emb)
        v = torch.cat([v, emb_proj[graph.batch]], dim=1)
        for layer in self.diffusion_step_encoder[1:]:
            v = layer(v)
        # Encode the edge irreps/features
        e = self.edge_encoder(
            torch.cat([
                graph.edge_attr,
                *[f for f in [graph.get('edge_cond')] if f is not None],
            ], dim=1)
        )
        # Propagate the scalar latent space (conditioned on c)
        v, _ = self.propagator(graph, v, e, emb)
        # Decode the latent node features
        v = self.node_decoder(v)
        # Return the output
        if self.learnable_variance:
            return torch.chunk(v, 2, dim=1)
        else:
            return v

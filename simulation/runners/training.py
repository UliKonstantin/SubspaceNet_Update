"""
Training components for simulations.

This module handles model creation and training.
"""

from typing import Dict, Any, Optional

class Trainer:
    """
    Handles model training for simulations.
    
    Responsible for:
    - Creating model based on configuration
    - Training model on dataset
    - Validating model performance
    """
    def __init__(self, config):
        self.config = config
        # Placeholder for actual implementation 

"""
Training implementation for trajectory-based DOA estimation.

This module provides training capabilities for working with trajectory datasets,
supporting step-by-step processing of trajectory data for improved sequential learning.
"""

import torch
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import copy

# Import from DCD_MUSIC package (no modifications)
from DCD_MUSIC.src.utils import device
from DCD_MUSIC.src.evaluation import evaluate_dnn_model

logger = logging.getLogger('SubspaceNet.training')

class TrainingConfig:
    """
    Configuration parameters for training.
    
    A simple container for training hyperparameters with attribute-based access
    and dictionary-like functionality.
    """
    def __init__(self, **kwargs):
        # Set default values
        self.learning_rate = 0.001
        self.weight_decay = 1e-9
        self.epochs = 50
        self.optimizer = "Adam"
        self.scheduler = "StepLR"
        self.step_size = 50
        self.gamma = 0.5
        self.batch_size = 32
        self.save_checkpoint = True
        self.checkpoint_path = Path("experiments/checkpoints")
        self.training_objective = "angle"  # Options: "angle", "range", "angle, range"
        
        # Update with provided parameters
        self.__dict__.update(kwargs)
        
    def __getitem__(self, key):
        return self.__dict__[key]
        
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def get(self, key, default=None):
        return self.__dict__.get(key, default)
        
    def update(self, new_params):
        self.__dict__.update(new_params)


class TrajectoryTrainer:
    """
    Trainer for trajectory-based DOA estimation models.
    
    This trainer handles time sequences where sources move along trajectories,
    processing each trajectory step sequentially. It supports both far-field
    and near-field scenarios.
    """
    
    def __init__(self, model, config: TrainingConfig, output_dir: Optional[Path] = None):
        """
        Initialize the trainer with a model and configuration.
        
        Args:
            model: Neural network model to train
            config: Training configuration parameters
            output_dir: Directory for saving results and checkpoints
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir or Path("experiments/results")
        
        # Training metrics and state
        self.best_model_wts = None
        self.min_valid_loss = float('inf')
        self.best_epoch = 0
        
        # Loss history
        self.train_losses = []
        self.valid_losses = []
        self.train_angles_losses = []
        self.train_ranges_losses = []
        self.valid_angles_losses = []
        self.valid_ranges_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        
        # Create output directories
        self._setup_directories()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
    def train(self, train_dataloader, valid_dataloader, seed=None):
        """
        Train the model using trajectory data.
        
        Args:
            train_dataloader: DataLoader for training data
            valid_dataloader: DataLoader for validation data
            seed: Random seed for reproducibility
            
        Returns:
            Trained model
        """
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Move model to device
        self.model = self.model.to(device)
        
        # Save best model weights
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        
        # Print training information
        self._print_training_info()
        
        # Training loop
        start_time = time.time()
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc, train_loss_components = self._train_epoch(train_dataloader, epoch)
            
            # Validation phase
            valid_loss, valid_acc, valid_loss_components = self._validate_epoch(valid_dataloader, epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()
                
            # Save history
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_accuracies.append(train_acc)
            self.valid_accuracies.append(valid_acc)
            
            # Save loss components if available
            if train_loss_components:
                angle_loss, range_loss = train_loss_components
                self.train_angles_losses.append(angle_loss)
                self.train_ranges_losses.append(range_loss)
                
            if valid_loss_components:
                angle_loss, range_loss = valid_loss_components
                self.valid_angles_losses.append(angle_loss)
                self.valid_ranges_losses.append(range_loss)
            
            # Report progress
            self._report_epoch_results(epoch, train_loss, train_acc, valid_loss, valid_acc, 
                                       train_loss_components, valid_loss_components)
            
            # Save best model
            if valid_loss < self.min_valid_loss:
                logger.info(f"Validation loss decreased ({self.min_valid_loss:.6f} → {valid_loss:.6f})")
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                
                if self.config.save_checkpoint:
                    self._save_checkpoint(f"best_{self.model.__class__.__name__}")
        
        # Training complete
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
        logger.info(f"Best validation loss: {self.min_valid_loss:.6f} at epoch {self.best_epoch+1}")
        
        # Restore best model weights
        self.model.load_state_dict(self.best_model_wts)
        
        # Plot training curves
        self._plot_training_curves()
        
        # Save final model
        if self.config.save_checkpoint:
            self._save_checkpoint(f"final_{self.model.__class__.__name__}")
        
        return self.model
    
    def _train_epoch(self, train_dataloader, epoch):
        """
        Train for one epoch using trajectory data.
        
        Args:
            train_dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, average accuracy, loss components)
        """
        self.model.train()
        
        total_loss = 0.0
        total_angle_loss = 0.0
        total_range_loss = 0.0
        total_accuracy = 0.0
        samples_count = 0
        
        # Loop over batches
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                            desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]")
        
        for batch_idx, batch_data in progress_bar:
            # Get batch data
            trajectories, sources_num, labels = batch_data
            batch_size, trajectory_length = trajectories.shape[0], trajectories.shape[1]
            
            # Track per-trajectory losses for this batch
            batch_losses = []
            batch_accuracies = []
            batch_angle_losses = []
            batch_range_losses = []
            
            # Loop over trajectory steps
            for step in range(trajectory_length):
                # Extract data for current step across all trajectories in batch
                step_data = trajectories[:, step].to(device)
                step_sources = sources_num[:, step].to(device)
                
                # Extract labels for current step
                # Labels shape depends on field type and objective:
                # - Far-field: angles only
                # - Near-field: angles and ranges
                step_labels = self._extract_step_labels(labels, step, step_sources)
                
                # Zero gradients
                self.optimizer.zero_grad()
                # reset gradients
                self.model.zero_grad()
                # Forward pass - this will vary based on model type and field type
                loss, accuracy, loss_components = self._forward_step(step_data, step_sources, step_labels)
                
                # Backward pass
                loss.backward(retain_graph=True)
                self.optimizer.step()
                # Store step results
                if batch_size > 0:
                    batch_losses.append(loss.item() / batch_size) # loss is sum over N_traj_in_batch items
                else:
                    batch_losses.append(0.0)
                
                if loss_components:
                    angle_loss, range_loss = loss_components
                    batch_angle_losses.append(angle_loss)
                    batch_range_losses.append(range_loss)
            
            # Calculate average loss and accuracy for this batch
            avg_batch_loss = np.mean(batch_losses)
            avg_batch_accuracy = np.mean(batch_accuracies)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{avg_batch_loss:.4f}",
                'acc': f"{avg_batch_accuracy*100:.1f}%"
            })
            
            # Update totals
            total_loss += avg_batch_loss * batch_size
            total_accuracy += avg_batch_accuracy * batch_size
            samples_count += batch_size
            
            if batch_angle_losses and batch_range_losses:
                total_angle_loss += np.mean(batch_angle_losses) * batch_size
                total_range_loss += np.mean(batch_range_losses) * batch_size
        
        # Calculate averages
        avg_loss = total_loss / samples_count
        avg_accuracy = total_accuracy / samples_count
        
        # Return loss components if available
        loss_components = None
        if total_angle_loss > 0 and total_range_loss > 0:
            avg_angle_loss = total_angle_loss / samples_count
            avg_range_loss = total_range_loss / samples_count
            loss_components = (avg_angle_loss, avg_range_loss)
            
        return avg_loss, avg_accuracy, loss_components
    
    def _validate_epoch(self, valid_dataloader, epoch):
        """
        Validate for one epoch using trajectory data.
        
        Args:
            valid_dataloader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, average accuracy, loss components)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_angle_loss = 0.0
        total_range_loss = 0.0
        total_accuracy = 0.0
        samples_count = 0
        
        # Loop over batches
        progress_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader),
                            desc=f"Epoch {epoch+1}/{self.config.epochs} [Valid]")
        
        with torch.no_grad():
            for batch_idx, batch_data in progress_bar:
                # Get batch data
                trajectories, sources_num, labels = batch_data
                batch_size, trajectory_length = trajectories.shape[0], trajectories.shape[1]
                
                # Track per-trajectory losses for this batch
                batch_losses = []
                batch_accuracies = []
                batch_angle_losses = []
                batch_range_losses = []
                
                # Loop over trajectory steps
                for step in range(trajectory_length):
                    # Extract data for current step across all trajectories in batch
                    step_data = trajectories[:, step].to(device)
                    step_sources = sources_num[:, step].to(device)
                    
                    # Extract labels for current step
                    step_labels = self._extract_step_labels(labels, step, step_sources)
                    
                    # Forward pass for validation
                    # loss_val is sum over batch_size items in step_data (from model.validation_step)
                    loss_val, accuracy_val, loss_components_val = self._forward_step(step_data, step_sources, step_labels, is_train=False)
                    
                    # Store per-item average loss for this step
                    if batch_size > 0:
                        batch_losses.append(loss_val.item() / batch_size) 
                    else:
                        batch_losses.append(0.0)
                    
                    batch_accuracies.append(accuracy_val) # Assuming accuracy_val is appropriately scaled or an average
                    
                    if loss_components_val: # If model.validation_step returns decomposed losses
                        angle_loss_step_sum, range_loss_step_sum = loss_components_val
                        if batch_size > 0:
                            # Assuming components are also sums that need averaging per item
                            batch_angle_losses.append(angle_loss_step_sum.item() / batch_size if hasattr(angle_loss_step_sum, 'item') else angle_loss_step_sum / batch_size)
                            batch_range_losses.append(range_loss_step_sum.item() / batch_size if hasattr(range_loss_step_sum, 'item') else range_loss_step_sum / batch_size)
                        else:
                            batch_angle_losses.append(0.0)
                            batch_range_losses.append(0.0)
                
                # Calculate average loss and accuracy for this batch
                avg_batch_loss = np.mean(batch_losses) if batch_losses else 0.0
                avg_batch_accuracy = np.mean(batch_accuracies) if batch_accuracies else 0.0
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{avg_batch_loss:.4f}",
                    'acc': f"{avg_batch_accuracy*100:.1f}%"
                })
                
                # Update totals (this logic remains the same as in _train_epoch)
                total_loss += avg_batch_loss * batch_size
                total_accuracy += avg_batch_accuracy * batch_size
                samples_count += batch_size
                
                if batch_angle_losses and batch_range_losses: # if components were successfully processed
                    total_angle_loss += np.mean(batch_angle_losses) * batch_size
                    total_range_loss += np.mean(batch_range_losses) * batch_size
        
        # Calculate averages
        avg_loss = total_loss / samples_count if samples_count > 0 else 0.0
        avg_accuracy = total_accuracy / samples_count if samples_count > 0 else 0.0
        
        # Return loss components if available
        loss_components = None
        if total_angle_loss > 0 or total_range_loss > 0: # Check if components were accumulated
            avg_angle_loss = total_angle_loss / samples_count if samples_count > 0 else 0.0
            avg_range_loss = total_range_loss / samples_count if samples_count > 0 else 0.0
            loss_components = (avg_angle_loss, avg_range_loss)
            
        return avg_loss, avg_accuracy, loss_components
    
    def _forward_step(self, step_data, step_sources, step_labels, is_train=True):
        """
        Perform a forward step through the model.
        
        This method handles both far-field and near-field scenarios and
        accommodates models with different training objectives.
        
        Args:
            step_data: Observation data for current step
            step_sources: Source counts for current step
            step_labels: Labels for current step
            is_train: Whether this is a training step
            
        Returns:
            Tuple of (loss, accuracy, loss_components)
        """
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(self.model, 'field_type') and self.model.field_type.lower() == "near"
        
        # Let PyTorch handle tensor types - don't modify tensor types here
        
        # Make sure step_sources is a single integer if needed for model forward pass
        # or use the batch of source counts for model's training/validation_step
        single_source_count_for_forward = step_sources[0].item() if step_sources.numel() > 1 else (step_sources.item() if step_sources.numel() == 1 else step_sources)

        if is_train:
            # --- Training Step ---
            if not is_near_field:
                angles = step_labels
                if hasattr(self.model, 'training_step'):
                    batch = (step_data, step_sources, angles)
                    loss, accuracy, eigen_regularization = self.model.training_step(batch, None)
                    return loss, accuracy, None  # Eigen_regularization is part of the loss from training_step
                else:
                    # Fallback if model doesn't have training_step (unlikely for SubspaceNet)
                    logger.warning_once("Model does not have a 'training_step' method. Using generic forward and MSE loss for training.")
                    angles_pred, source_estimation, _ = self.model(step_data, single_source_count_for_forward)
                    loss = self._calculate_loss(angles_pred, angles)
                    accuracy = self._calculate_accuracy(step_sources, source_estimation)
                    return loss, accuracy, None
            else: # Near-field training
                angles, ranges = step_labels
                if hasattr(self.model, 'training_step'):
                    # DCD_MUSIC SubspaceNet expects labels as [batch, num_sources * 2] for near-field training_step
                    model_step_labels = torch.cat([angles, ranges], dim=1) 
                    batch = (step_data, step_sources, model_step_labels)
                    loss, accuracy, eigen_regularization = self.model.training_step(batch, None)
                    
                    # Check if loss is returned as tuple (combined, angle, range) by model's training_step
                    if isinstance(loss, tuple) and len(loss) == 3:
                        combined_loss, angle_loss_val, range_loss_val = loss
                        return combined_loss, accuracy, (angle_loss_val, range_loss_val) # loss_components might be tensors
                    else:
                        return loss, accuracy, None # Eigen_regularization is part of the loss from training_step
                else:
                    # Fallback if model doesn't have training_step
                    logger.warning_once("Model does not have a 'training_step' method. Using generic forward and MSE loss for near-field training.")
                    angles_pred, ranges_pred, source_estimation, _ = self.model(step_data, single_source_count_for_forward)
                    angle_loss = self._calculate_loss(angles_pred, angles)
                    range_loss = self._calculate_loss(ranges_pred, ranges)
                    combined_loss = angle_loss + range_loss
                    accuracy = self._calculate_accuracy(step_sources, source_estimation)
                    return combined_loss, accuracy, (angle_loss.item(), range_loss.item())
        else:
            # --- Validation Step ---
            if hasattr(self.model, 'validation_step'):
                model_step_labels = None
                if not is_near_field:
                    angles = step_labels
                    model_step_labels = angles
                else: # Near-field validation
                    angles, ranges = step_labels
                    # DCD_MUSIC SubspaceNet expects labels as [batch, num_sources * 2] for near-field validation_step
                    model_step_labels = torch.cat([angles, ranges], dim=1)
                
                batch = (step_data, step_sources, model_step_labels)
                # SubspaceNet.validation_step typically returns (loss, acc, eigen_reg_value)
                # The loss here should be the primary loss criterion (e.g., RMSPE) without regularization added for metrics.
                loss, accuracy = self.model.validation_step(batch, None) # We don't use the eigen_reg for the loss metric

                # If model.validation_step provides decomposed losses (e.g., for near-field)
                loss_components = None
                if isinstance(loss, tuple) and len(loss) == 3: # Assuming (combined, angle, range) like training_step could
                    combined_loss, angle_loss_val, range_loss_val = loss
                    loss = combined_loss # Use combined loss for overall validation loss
                    loss_components = (angle_loss_val.item() if isinstance(angle_loss_val, torch.Tensor) else angle_loss_val, 
                                       range_loss_val.item() if isinstance(range_loss_val, torch.Tensor) else range_loss_val)
                elif isinstance(loss, torch.Tensor) and is_near_field: # If only combined loss is returned, no components
                     pass # loss is already the combined loss

                return loss, accuracy, loss_components

            else: # Fallback if model doesn't have validation_step
                logger.warning_once("Model does not have a 'validation_step' method. Using generic forward and MSE loss for validation.")
                if not is_near_field:
                    angles = step_labels
                    angles_pred, source_estimation, _ = self.model(step_data, single_source_count_for_forward)
                    loss = self._calculate_loss(angles_pred, angles)
                    accuracy = self._calculate_accuracy(step_sources, source_estimation)
                    return loss, accuracy, None
                else: # Near-field validation with MSE fallback
                    angles, ranges = step_labels
                    angles_pred, ranges_pred, source_estimation, _ = self.model(step_data, single_source_count_for_forward)
                    angle_loss = self._calculate_loss(angles_pred, angles)
                    range_loss = self._calculate_loss(ranges_pred, ranges)
                    combined_loss = angle_loss + range_loss
                    accuracy = self._calculate_accuracy(step_sources, source_estimation)
                    return combined_loss, accuracy, (angle_loss.item(), range_loss.item())
    
    def _extract_step_labels(self, labels, step, step_sources):
        """
        Extract labels for the current step.
        
        Args:
            labels: Full labels tensor [batch_size, trajectory_length, max_sources]
            step: Current time step
            step_sources: Number of sources for current step
            
        Returns:
            Labels for current step
        """
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(self.model, 'field_type') and self.model.field_type.lower() == "near"
        
        # Get labels for the current step across all batch items
        step_labels = labels[:, step].to(device)
        
        if not is_near_field:
            # For far-field, we only need angles
            return step_labels
        else:
            # For near-field, we need to split into angles and ranges
            batch_size = step_labels.shape[0]
            angles = []
            ranges = []
            
            for i in range(batch_size):
                # Get number of sources for this item
                num_sources = step_sources[i].item()
                
                # Split labels for this example - simplest approach
                example_angles = step_labels[i, :num_sources]
                example_ranges = step_labels[i, num_sources:num_sources*2]
                
                angles.append(example_angles)
                ranges.append(example_ranges)
            
            # Stack back to batch - let PyTorch handle tensor types
            angles = torch.stack(angles)
            ranges = torch.stack(ranges)
            
            return (angles, ranges)
    
    def _calculate_loss(self, predictions, targets):
        """
        Calculate loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Loss value
        """
        # For now, use a simple MSE loss, but this can be replaced with
        # more sophisticated losses from the original DCD-MUSIC implementation
        return torch.nn.functional.mse_loss(predictions, targets)
    
    def _calculate_accuracy(self, true_sources, estimated_sources):
        """
        Calculate accuracy of source count estimation.
        
        Args:
            true_sources: True number of sources
            estimated_sources: Estimated number of sources
            
        Returns:
            Accuracy as a fraction (0-1)
        """
        if estimated_sources is None:
            return 1.0  # If model doesn't estimate sources, assume perfect
            
        # Calculate accuracy
        correct = (estimated_sources == true_sources).sum().item()
        total = true_sources.size(0)
        
        return correct / total if total > 0 else 1.0
    
    def _report_epoch_results(self, epoch, train_loss, train_acc, valid_loss, valid_acc, 
                             train_loss_components, valid_loss_components):
        """
        Report results after each epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            valid_loss: Validation loss
            valid_acc: Validation accuracy
            train_loss_components: Training loss components (angle, range)
            valid_loss_components: Validation loss components (angle, range)
        """
        report = f"[Epoch {epoch+1}/{self.config.epochs}] "
        report += f"Train loss: {train_loss:.6f}, Valid loss: {valid_loss:.6f}, "
        report += f"Train acc: {train_acc*100:.2f}%, Valid acc: {valid_acc*100:.2f}%"
        
        # Add loss components if available
        if train_loss_components and valid_loss_components:
            train_angle, train_range = train_loss_components
            valid_angle, valid_range = valid_loss_components
            report += f"\nAngle loss: Train={train_angle:.6f}, Valid={valid_angle:.6f}, "
            report += f"Range loss: Train={train_range:.6f}, Valid={valid_range:.6f}"
        
        # Log learning rate
        lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
        report += f"\nLearning rate: {lr:.6e}"
        
        logger.info(report)
    
    def _create_optimizer(self):
        """
        Create optimizer based on configuration.
        
        Returns:
            PyTorch optimizer
        """
        optimizer_type = self.config.optimizer
        
        if optimizer_type == "Adam":
            return optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_type == "SGD":
            return optim.SGD(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif optimizer_type == "RMSprop":
            return optim.RMSprop(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            logger.warning(f"Unsupported optimizer: {optimizer_type}, using Adam instead")
            return optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler based on configuration.
        
        Returns:
            PyTorch scheduler
        """
        scheduler_type = self.config.scheduler
        
        if scheduler_type == "StepLR":
            return lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif scheduler_type == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.gamma,
                patience=10,
                verbose=True
            )
        elif scheduler_type == "CosineAnnealingLR":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        else:
            logger.warning(f"Unsupported scheduler: {scheduler_type}, using StepLR instead")
            return lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
    
    def _setup_directories(self):
        """Create necessary directories for saving results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def _save_checkpoint(self, name):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_accuracies': self.train_accuracies,
            'valid_accuracies': self.valid_accuracies,
            'config': self.config.__dict__
        }, path)
        
        logger.info(f"Model checkpoint saved to {path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plots_dir / f"loss_curve_{timestamp}.png")
        
        # Plot accuracy curves
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(self.train_accuracies) * 100, label='Training Accuracy')
        plt.plot(np.array(self.valid_accuracies) * 100, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plots_dir / f"accuracy_curve_{timestamp}.png")
        
        # Plot angle and range losses if available
        if self.train_angles_losses and self.valid_angles_losses:
            # Angle loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_angles_losses, label='Training Angle Loss')
            plt.plot(self.valid_angles_losses, label='Validation Angle Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Angle Loss')
            plt.title('Training and Validation Angle Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.plots_dir / f"angle_loss_curve_{timestamp}.png")
            
            # Range loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_ranges_losses, label='Training Range Loss')
            plt.plot(self.valid_ranges_losses, label='Validation Range Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Range Loss')
            plt.title('Training and Validation Range Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.plots_dir / f"range_loss_curve_{timestamp}.png")
    
    def _print_training_info(self):
        """Print training configuration and model information."""
        logger.info("Starting trajectory-based training")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {device}")
        logger.info(f"Training objective: {self.config.training_objective}")
        logger.info(f"Optimizer: {self.config.optimizer}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Weight decay: {self.config.weight_decay}")
        logger.info(f"Scheduler: {self.config.scheduler}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Epochs: {self.config.epochs}")
        
        # Print model size (number of parameters)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_params:,}") 
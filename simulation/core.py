"""
Core simulation components and main controller.

This module contains the main Simulation class that orchestrates
the simulation process from data handling to evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import torch
from tqdm import tqdm
import datetime

from config.schema import Config
from .runners.data import TrajectoryDataHandler
from .runners.training import Trainer, TrainingConfig, TrajectoryTrainer
from .runners.evaluation import Evaluator
from simulation.kalman_filter import KalmanFilter1D, BatchKalmanFilter1D
from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
from DCD_MUSIC.src.signal_creation import Samples
from DCD_MUSIC.src.evaluation import get_model_based_method, evaluate_model_based

logger = logging.getLogger(__name__)

# Device setup for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Simulation:
    """
    Main simulation controller.
    
    Handles the orchestration of data preparation, model training,
    and evaluation for both single runs and parametric scenarios.
    """
    def __init__(self, config: Config, components: Dict[str, Any], output_dir: Optional[Path] = None):
        """
        Initialize the simulation controller.
        
        Args:
            config: Configuration object with simulation parameters
            components: Dictionary of pre-created components
            output_dir: Directory for saving results
        """
        self.config = config
        self.components = components
        self.output_dir = output_dir or Path("experiments/results")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize handlers from components
        self.system_model = components.get("system_model")
        self.model = components.get("model")
        self.trajectory_handler = components.get("trajectory_handler")
        
        # Create trajectory handler if needed
        if self.config.trajectory.enabled and self.trajectory_handler is None:
            self._create_trajectory_handler()
        
        # Initialize result containers
        self.dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.trained_model = None
        self.results = {}
        
        logger.info(f"Simulation initialized with output directory: {self.output_dir}")
        logger.info(f"Trajectory mode: {'Enabled' if self.config.trajectory.enabled else 'Disabled'}")
        
    def run(self) -> Dict[str, Any]:
        """
        Run a single simulation.
        
        Returns:
            Dict containing evaluation results.
        """
        logger.info("Starting simulation")
        
        try:
            # Execute data pipeline
            self._run_data_pipeline()
            
            # Check if data pipeline was successful
            if self.train_dataloader is None:
                logger.error("Data pipeline failed, skipping training and evaluation")
                return {"status": "error", "message": "Data pipeline failed"}
            
            # Load model if configured
            if self.config.simulation.load_model:
                if hasattr(self.config.simulation, 'model_path') and self.config.simulation.model_path:
                    model_path = Path(self.config.simulation.model_path)
                    success, message = self._load_and_apply_weights(model_path, device)
                    if not success:
                        # Loading failed, return error status
                        return {"status": "error", "message": message}
                    # If partial success, message is logged by helper, continue simulation
                else:
                    logger.error("Model loading requested but no model_path provided in config")
                    return {"status": "error", "message": "No model_path provided"}
            
            # Execute training pipeline if enabled AND train_model is True
            if self.config.training.enabled and self.config.simulation.train_model:
                logger.info("Running training pipeline (training.enabled=True, simulation.train_model=True)")
                self._run_training_pipeline()
                
                # Check if training was successful
                if self.trained_model is None:
                    logger.error("Training pipeline failed, skipping evaluation")
                    return {"status": "error", "message": "Training pipeline failed"}
            elif not self.config.simulation.train_model:
                logger.info("Skipping training (simulation.train_model=False)")
            elif not self.config.training.enabled:
                logger.info("Skipping training (training.enabled=False)")
                    
            # Execute evaluation pipeline if enabled
            if self.config.simulation.evaluate_model:
                # Make sure we have a model to evaluate
                if self.trained_model is None and self.model is not None:
                    logger.info("Using non-trained model for evaluation")
                    self.trained_model = self.model
                
                # Run the evaluation
                if self.trained_model is not None:
                    self._run_evaluation_pipeline()
                else:
                    logger.error("No model available for evaluation")
                    return {"status": "error", "message": "No model available for evaluation"}
                
            # Execute online learning if enabled
            if hasattr(self.config, 'online_learning') and getattr(self.config.online_learning, 'enabled', False):
                logger.info("Running online learning pipeline (online_learning.enabled=True)")
                
                # Make sure we have a model for online learning
                if self.trained_model is None and self.model is not None:
                    logger.info("Using non-trained model for online learning")
                    self.trained_model = self.model
                
                # Run online learning
                if self.trained_model is not None:
                    self._run_online_learning()
                else:
                    logger.error("No model available for online learning")
                    return {"status": "error", "message": "No model available for online learning"}
                
            # Save model if configured
            if self.config.simulation.save_model and self.trained_model is not None:
                self._save_model_state(self.trained_model)
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.exception(f"Error running simulation: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}
    
    def _run_data_pipeline(self) -> None:
        """
        Execute the data preparation pipeline.
        
        Creates or loads datasets based on configuration and
        sets up dataloaders for training, validation, and testing.
        """
        logger.info("Starting data pipeline")
        
        # Check if trajectory mode is enabled
        if self.config.trajectory.enabled:
            if self.trajectory_handler:
                logger.info("Using trajectory-based data pipeline")
                dataset, samples_model = self._create_trajectory_dataset()
                # Store samples_model for potential use in evaluation
                self.components["samples_model"] = samples_model
            else:
                logger.error("Trajectory mode enabled but no trajectory_handler found")
                return
        else:
            # Use standard dataset creation from factory
            if "dataset" in self.components:
                logger.info("Using pre-created dataset from components")
                dataset = self.components["dataset"]
            elif self.config.dataset.create_data:
                logger.info("Creating new dataset")
                dataset = self._create_standard_dataset()
            else:
                logger.info("Loading existing dataset")
                dataset = self._load_standard_dataset()
        
        if dataset is None:
            logger.error("Failed to create or load dataset")
            return
        
        self.dataset = dataset
        
        # Create dataloaders for training and validation
        logger.info("Creating dataloaders with batch size: %d", self.config.training.batch_size)
        self.train_dataloader, self.valid_dataloader = dataset.get_dataloaders(
            batch_size=self.config.training.batch_size,
            validation_split=self.config.dataset.validation_split if hasattr(self.config.dataset, "validation_split") else 0.2
        )
        
        logger.info(f"Created train dataloader with {len(self.train_dataloader)} batches")
        logger.info(f"Created validation dataloader with {len(self.valid_dataloader)} batches")
        
        # Store dataloaders in components
        self.components["train_dataloader"] = self.train_dataloader
        self.components["valid_dataloader"] = self.valid_dataloader
        
        # Create test dataset (1/4 of training dataset size)
        logger.info("Creating test dataset (1/4 of training size)")
        test_samples_size = self.config.dataset.samples_size // 4
        
        if self.config.trajectory.enabled:
            if self.trajectory_handler:
                test_dataset, _ = self._create_trajectory_dataset_for_testing(test_samples_size)
            else:
                logger.error("Trajectory mode enabled but no trajectory_handler found for testing")
                return
        else:
            # Create standard test dataset
            test_dataset = self._create_standard_test_dataset(test_samples_size)
        
        if test_dataset is None:
            logger.error("Failed to create test dataset")
            return
            
        # Create test dataloader directly
        from torch.utils.data import DataLoader
        self.test_dataloader = DataLoader(
            test_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=False,  # No shuffling for test data
            # Add collate_fn if needed, especially for trajectory data
            collate_fn=test_dataset._collate_trajectories if hasattr(test_dataset, '_collate_trajectories') else None
        )
        
        logger.info(f"Created test dataloader with {len(self.test_dataloader)} batches")
        
        # Store test dataloader in components
        self.components["test_dataloader"] = self.test_dataloader
        
    def _create_trajectory_dataset(self) -> Tuple[Any, Any]:
        """Create a dataset with trajectory support."""
        return self.trajectory_handler.create_dataset(
            samples_size=self.config.dataset.samples_size,
            trajectory_length=self.config.trajectory.trajectory_length,
            trajectory_type=self.config.trajectory.trajectory_type,
            save_dataset=self.config.trajectory.save_trajectory,
            dataset_path=Path("data/datasets").absolute()
        )
    
    def _create_standard_dataset(self) -> Any:
        """Create a standard (non-trajectory) dataset."""
        # This would normally call the factory's create_dataset function
        # For now, we'll just use what's in components if available
        if "dataset" in self.components:
            return self.components["dataset"]
        
        # If we need to implement dataset creation directly:
        from DCD_MUSIC.src.signal_creation import Samples
        from DCD_MUSIC.src.data_handler import create_dataset
        
        samples_model = Samples(self.system_model.params)
        dataset, _ = create_dataset(
            samples_model=samples_model,
            samples_size=self.config.dataset.samples_size,
            save_datasets=self.config.dataset.save_dataset,
            datasets_path=Path("data/datasets").absolute(),
            true_doa=self.config.dataset.true_doa_train,
            true_range=self.config.dataset.true_range_train,
            phase="train"
        )
        return dataset
    
    def _load_standard_dataset(self) -> Any:
        """Load a standard (non-trajectory) dataset."""
        # Implementation depends on the dataset storage format
        from DCD_MUSIC.src.data_handler import load_datasets
        
        try:
            dataset = load_datasets(
                system_model_params=self.system_model.params,
                samples_size=self.config.dataset.samples_size,
                datasets_path=Path("data/datasets").absolute(),
                is_training=True
            )
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def _run_training_pipeline(self) -> None:
        """Execute the training pipeline with trajectory support."""
        logger.info("Starting training pipeline")
        
        # Determine if we should use trajectory-based training
        use_trajectory_training = self.config.trajectory.enabled
        
        # Create TrainingConfig
        training_config = TrainingConfig(
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            optimizer=self.config.training.optimizer,
            scheduler=self.config.training.scheduler,
            step_size=self.config.training.step_size,
            gamma=self.config.training.gamma,
            training_objective=self.config.training.training_objective,
            save_checkpoint=getattr(self.config.training, "save_checkpoint", True),
            checkpoint_path=self.output_dir / "checkpoints"
        )
        
        # Create appropriate trainer based on dataset type
        if use_trajectory_training:
            logger.info("Using trajectory-based trainer")
            trainer = TrajectoryTrainer(
                model=self.model,
                config=training_config,
                output_dir=self.output_dir
            )
        else:
            # Use standard trainer from DCD_MUSIC if not trajectory-based
            logger.info("Using standard trainer")
            if "trainer" in self.components:
                logger.info("Using pre-created trainer from components")
                trainer = self.components["trainer"]
            else:
                # Import Trainer from DCD_MUSIC
                from DCD_MUSIC.src.training import Trainer as DCDTrainer, TrainingParamsNew
                
                # Convert our config to DCD's TrainingParams format
                training_params = TrainingParamsNew(
                    learning_rate=training_config.learning_rate,
                    weight_decay=training_config.weight_decay,
                    epochs=training_config.epochs,
                    optimizer=training_config.optimizer,
                    scheduler=training_config.scheduler,
                    step_size=training_config.step_size,
                    gamma=training_config.gamma,
                    training_objective=training_config.training_objective,
                    batch_size=training_config.batch_size
                )
                
                trainer = DCDTrainer(
                    model=self.model,
                    training_params=training_params,
                    show_plots=False
                )
        
        # Store trainer in components for potential reuse
        self.components["trainer"] = trainer
        
        # Train the model
        logger.info(f"Starting model training with {training_config.epochs} epochs")
        self.trained_model = trainer.train(
            self.train_dataloader,
            self.valid_dataloader,
            seed=42  # For reproducibility
        )
        
        # Store the trained model back in components
        self.components["model"] = self.trained_model
        
        logger.info("Training completed")
        
    def _run_evaluation_pipeline(self) -> None:
        """
        Execute the evaluation pipeline.

        This function serves as the main wrapper for the evaluation process:
        1. Evaluates the DNN model with Kalman filtering
        2. Evaluates classic subspace methods (if configured)
        3. Logs comparative results
        """
        logger.info("Starting evaluation pipeline")
        
        # Validate that we have a model to evaluate
        if self.trained_model is None and self.model is not None:
            logger.info("Using loaded model for evaluation")
            self.trained_model = self.model
        elif self.trained_model is None:
            logger.error("No model available for evaluation")
            return
        
        # Ensure model is in evaluation mode
        self.trained_model.eval()
        
        # Check if we have test dataloader
        if not hasattr(self, 'test_dataloader') or self.test_dataloader is None:
            if self.valid_dataloader is not None:
                logger.warning("No test dataloader available, using validation dataloader instead")
                test_dataloader = self.valid_dataloader
            else:
                logger.error("No validation or test dataloader available for evaluation")
                return
        else:
            test_dataloader = self.test_dataloader
        
        try:
            logger.info("Evaluating model(s) on trajectory test data")
            
            # Check if near-field mode is active
            is_near_field = hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near"
            if is_near_field:
                error_msg = "Near-field option is not available in the current evaluation pipeline"
                logger.error(error_msg)
                self.results["evaluation_error"] = error_msg
                return
            
            # Get Kalman filter parameters from config
            kf_Q, kf_R, kf_P0 = KalmanFilter1D.from_config(self.config)
            
            # Instantiate RMSPELoss criterion
            rmspe_criterion = RMSPELoss().to(device)
            
            # Result containers
            dnn_trajectory_results = []  # Store DNN+KF results
            
            # Overall metrics
            dnn_total_loss = 0.0
            dnn_total_samples = 0
            classic_methods_losses = {}  # Dictionary to store losses for each classic method
            
            # Get list of classic subspace methods to evaluate
            classic_methods = []
            if hasattr(self.config.simulation, 'subspace_methods') and self.config.simulation.subspace_methods:
                classic_methods = self.config.simulation.subspace_methods
                logger.info(f"Will evaluate classic subspace methods: {classic_methods}")
                
                # Initialize loss accumulators for each method
                for method in classic_methods:
                    classic_methods_losses[method] = {"total_loss": 0.0, "total_samples": 0}
            
            # Trajectory-level evaluation with both DNN+KF and classic methods
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Evaluating trajectories")):
                    # Get batch data
                    trajectories, sources_num, labels = batch_data
                    batch_size, trajectory_length = trajectories.shape[0], trajectories.shape[1]
                    
                    # Initialize storage for batch results
                    batch_dnn_preds = [[] for _ in range(batch_size)]
                    batch_dnn_kf_preds = [[] for _ in range(batch_size)]
                    
                    # Find maximum number of sources in batch for efficient batch processing
                    max_sources = torch.max(sources_num).item()
                    
                    # Initialize the batch Kalman filter
                    batch_kf = BatchKalmanFilter1D.from_config(
                        self.config,
                        batch_size=batch_size,
                        max_sources=max_sources,
                        device=device
                    )
                    
                    # Initialize states directly from first step ground truth
                    # Each trajectory might have different number of sources
                    batch_kf.initialize_states(labels[:, 0, :max_sources], sources_num[:, 0])
                    
                    # Process each time step in the trajectory sequentially (required for Kalman filtering)
                    for step in range(trajectory_length):
                        # Get data for this time step across all trajectories
                        step_data = trajectories[:, step].to(device)
                        step_sources = sources_num[:, step].to(device)
                        
                        # Create source mask for this step
                        step_mask = torch.arange(max_sources, device=device).expand(batch_size, -1) < step_sources[:, None]
                        
                        # Evaluate DNN model and update Kalman filters for this time step
                        model_preds, kf_preds, step_loss = self._evaluate_dnn_model_kf_step_batch(
                            step_data, step_sources, labels[:, step, :max_sources], step_mask, batch_kf, 
                            rmspe_criterion, is_near_field
                        )
                        
                        # Accumulate loss and store predictions
                        dnn_total_loss += step_loss
                        dnn_total_samples += batch_size
                        
                        # Store predictions for this time step
                        for i in range(batch_size):
                            batch_dnn_preds[i].append(model_preds[i])
                            batch_dnn_kf_preds[i].append(kf_preds[i])
                        
                        # Evaluate classic methods if enabled
                        if classic_methods:
                            classic_results = self._evaluate_classic_methods_step_batch(
                                step_data, step_sources, labels[:, step, :max_sources],
                                rmspe_criterion, classic_methods
                            )
                            
                            # Update classic method losses
                            for method, results in classic_results.items():
                                classic_methods_losses[method]["total_loss"] += results["total_loss"]
                                classic_methods_losses[method]["total_samples"] += results["count"]
                    
                    # Store complete trajectory results
                    for i in range(batch_size):
                        num_sources_array = sources_num[i].cpu().numpy()
                        gt_trajectory = np.array([
                            labels[i, t, :num_sources_array[t]].cpu().numpy()
                            for t in range(trajectory_length)
                        ])
                        
                        dnn_trajectory_results.append({
                            'model_predictions': batch_dnn_preds[i],
                            'kf_predictions': batch_dnn_kf_preds[i],
                            'ground_truth': gt_trajectory,
                            'sources': num_sources_array
                        })
                
                # Log evaluation results
                self._log_evaluation_results(
                    dnn_total_loss, 
                    dnn_total_samples,
                    classic_methods_losses,
                    dnn_trajectory_results,
                    None  # No classic trajectory results to log
                )
                
                # Store DNN results
                self.results["dnn_trajectory_results"] = dnn_trajectory_results
                    
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            self.results["evaluation_error"] = str(e)

    def _evaluate_dnn_model_kf_step_batch(
        self, 
        step_data: torch.Tensor, 
        step_sources: torch.Tensor,
        step_angles: torch.Tensor,
        step_mask: torch.Tensor,
        batch_kf: BatchKalmanFilter1D,
        rmspe_criterion: RMSPELoss,
        is_near_field: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Evaluate the DNN model for a batch of steps and apply Kalman filtering.
        
        Args:
            step_data: Batch of trajectory step data for current time step [batch_size, T, N]
            step_sources: Number of sources for each trajectory at current step [batch_size]
            step_angles: Ground truth angles tensor [batch_size, max_sources]
            step_mask: Mask indicating valid angles for each trajectory at current step
            batch_kf: BatchKalmanFilter1D instance for efficient batch processing
            rmspe_criterion: Loss criterion for evaluation
            is_near_field: Whether the model is for near-field estimation
            
        Returns:
            Tuple containing:
            - List of model predictions for each trajectory
            - List of KF predictions for each trajectory
            - Total loss for this batch at this step
        """
        batch_size = step_data.shape[0]
        max_sources = batch_kf.max_sources
        
        # Collect model predictions per trajectory
        batch_angles_pred_list = []
        total_loss = 0.0

        # Loop through batch because the model (or underlying method) doesn't support batch source counts
        for i in range(batch_size):
            # Extract data for a single trajectory
            single_step_data = step_data[i].unsqueeze(0) # Add batch dim back
            single_step_sources = step_sources[i] # This should be a scalar tensor
            num_sources_item = single_step_sources.item()

            # Skip if no sources
            if num_sources_item <= 0:
                batch_angles_pred_list.append(torch.empty((0,), device=device))
                continue

            # Ensure source count is a 0-dim tensor or scalar int if needed by model
            # Some models might expect tensor(2), others just 2.
            # Assuming the model can handle a 0-dim tensor:
            # single_step_sources = single_step_sources # Keep as 0-dim tensor
            # Or if it needs an int:
            # single_step_sources_arg = num_sources_item

            # Model forward pass for single trajectory
            # We pass the 0-dim tensor, assuming the model handles it.
            # Adjust if the model strictly requires an integer.
            if is_near_field:
                # TODO: Confirm near-field output structure and handling if needed
                angles_pred_single, _, _ = self.trained_model(single_step_data, single_step_sources)
            else:
                angles_pred_single, _, _ = self.trained_model(single_step_data, single_step_sources)

            # Ensure prediction tensor has correct shape [1, num_sources]
            angles_pred_single = angles_pred_single.view(1, -1)[:, :num_sources_item]
            batch_angles_pred_list.append(angles_pred_single.squeeze(0))

            # Calculate loss for this trajectory
            with torch.no_grad():
                truth = step_angles[i, :num_sources_item].unsqueeze(0)
                loss = rmspe_criterion(angles_pred_single, truth)
                total_loss += loss.item()

        # Combine predictions into a batch tensor for KF update
        # Need padding if source counts vary
        padded_angles_pred = torch.zeros((batch_size, max_sources), device=device)
        for i, pred in enumerate(batch_angles_pred_list):
            num_sources = pred.shape[0]
            if num_sources > 0:
                padded_angles_pred[i, :num_sources] = pred

        # Apply the Kalman filter to the predicted angles (using the padded batch tensor)
        # First, get predictions (before update) for output
        kf_predictions_before_update = batch_kf.predict().cpu().numpy()

        # Then update with new measurements (model predictions)
        batch_kf.update(padded_angles_pred, step_mask)

        # Convert predictions to list of numpy arrays with proper dimensions
        model_predictions_list = []
        kf_predictions_list = []

        for i in range(batch_size):
            num_sources = step_sources[i].item()
            if num_sources > 0:
                model_predictions_list.append(padded_angles_pred[i, :num_sources].cpu().numpy())
                kf_predictions_list.append(kf_predictions_before_update[i, :num_sources])
            else:
                model_predictions_list.append(np.array([]))
                kf_predictions_list.append(np.array([]))

        return model_predictions_list, kf_predictions_list, total_loss

    def _evaluate_classic_methods_step_batch(
        self,
        step_data: torch.Tensor,
        step_sources: torch.Tensor,
        step_angles: torch.Tensor,
        rmspe_criterion: RMSPELoss,
        methods: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate classic subspace methods for a batch of trajectories at one time step.
        
        Args:
            step_data: Batch of trajectory step data for current time step [batch_size, T, N]
            step_sources: Number of sources for each trajectory at current step [batch_size]
            step_angles: Tensor of ground truth angles for each trajectory at current step
            rmspe_criterion: Loss criterion for evaluation
            methods: List of classic subspace methods to evaluate
            
        Returns:
            Dictionary mapping method names to their batch results
        """
        batch_size = step_data.shape[0]
        results = {}
        
        # Initialize results for all methods
        for method in methods:
            results[method] = {"total_loss": 0.0, "count": 0}
        
        # Process each trajectory separately for classic methods
        # (classic methods generally don't support batch processing)
        for traj_idx in range(batch_size):
            # Get data for this trajectory at this time step
            traj_data = step_data[traj_idx]
            traj_sources = step_sources[traj_idx]
            num_sources = traj_sources.item()
            
            # Skip trajectories with no sources
            if num_sources <= 0:
                continue
                
            # Extract the ground truth angles for this trajectory
            traj_angles = step_angles[traj_idx, :num_sources]
            
            try:
                # Get system model parameters
                system_model_params = self.system_model.params
                
                # Store original M and temporarily update to current sources
                original_M = system_model_params.M
                system_model_params.M = int(num_sources)
                
                # Format data for test_step
                # Transform to [1, N, T] as expected by classic methods
                processed_data = traj_data.unsqueeze(0)
                
                # Create batch data structure for test_step
                batch_data = (
                    processed_data,  # Shape: [1, N, T]
                    traj_sources.unsqueeze(0),  # Shape: [1]
                    traj_angles.unsqueeze(0),  # Shape: [1, num_sources]
                    torch.ones_like(traj_angles).unsqueeze(0)  # Shape: [1, num_sources]
                )
                
                # Evaluate each classic method
                for method_name in methods:
                    try:
                        # Get method instance
                        classic_method = get_model_based_method(method_name, system_model_params)
                        
                        if classic_method is not None:
                            # Call test_step to evaluate this method
                            loss, accuracy, _ = classic_method.test_step(batch_data, 0)
                            
                            # Accumulate results
                            results[method_name]["total_loss"] += loss
                            results[method_name]["count"] += 1
                        else:
                            logger.warning(f"Failed to initialize classic method: {method_name}")
                    except Exception as method_error:
                        logger.error(f"Error with method {method_name} for trajectory {traj_idx}: {method_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Restore original M
                system_model_params.M = original_M
                    
            except Exception as e:
                logger.error(f"Error evaluating classic methods for trajectory {traj_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        return results

    def _log_evaluation_results(
        self,
        dnn_total_loss: float,
        dnn_total_samples: int,
        classic_methods_losses: Dict[str, Dict[str, float]],
        dnn_trajectory_results: List[Dict[str, Any]],
        classic_trajectory_results: List[Dict[str, Any]]
    ) -> None:
        """
        Log the evaluation results for both DNN and classic methods.
        
        Args:
            dnn_total_loss: Accumulated loss for DNN model
            dnn_total_samples: Total samples evaluated for DNN
            classic_methods_losses: Dictionary of losses for classic methods
            dnn_trajectory_results: List of trajectory results for DNN
            classic_trajectory_results: List of trajectory results for classic methods
        """
        # Calculate average loss for DNN model
        dnn_avg_loss = dnn_total_loss / max(dnn_total_samples, 1)
        
        # Log DNN results
        logger.info(f"DNN Model - Average loss: {dnn_avg_loss:.6f}")
        
        # Calculate and log average losses for classic methods
        classic_methods_avg_losses = {}
        for method, loss_data in classic_methods_losses.items():
            if loss_data["total_samples"] > 0:
                avg_loss = loss_data["total_loss"] / loss_data["total_samples"]
                classic_methods_avg_losses[method] = avg_loss
                logger.info(f"Classic Method {method} - Average loss: {avg_loss:.6f}")
        
        # Store metrics in results
        self.results["dnn_test_loss"] = dnn_avg_loss
        self.results["classic_methods_test_losses"] = classic_methods_avg_losses
        
        # Compare DNN vs classic methods if both are available
        if classic_methods_avg_losses:
            logger.info("Comparative Results:")
            for method, avg_loss in classic_methods_avg_losses.items():
                diff = dnn_avg_loss - avg_loss
                relative_diff = diff / avg_loss * 100 if avg_loss != 0 else float('inf')
                
                if diff < 0:
                    logger.info(f"DNN outperforms {method} by {abs(diff):.6f} ({abs(relative_diff):.2f}%)")
                else:
                    logger.info(f"{method} outperforms DNN by {diff:.6f} ({relative_diff:.2f}%)")
        
        # Log total number of trajectories evaluated
        logger.info(f"Evaluated {len(dnn_trajectory_results)} trajectories with DNN model")
        if classic_trajectory_results:
            logger.info(f"Evaluated {len(classic_trajectory_results)} trajectories with classic methods")
            
        # Add print statements for detailed evaluation results
        print("\n" + "="*80)
        print(f"{'EVALUATION RESULTS':^80}")
        print("="*80)
        
        # Print DNN results
        print(f"\n{'DNN MODEL WITH KALMAN FILTER':^80}")
        print("-"*80)
        print(f"Average Loss: {dnn_avg_loss:.6f}")
        print(f"Total Samples: {dnn_total_samples}")
        print(f"Trajectories Evaluated: {len(dnn_trajectory_results)}")
        
        # Print classic methods results if available
        if classic_methods_avg_losses:
            print(f"\n{'CLASSIC SUBSPACE METHODS':^80}")
            print("-"*80)
            print(f"{'Method':<20} {'Average Loss':<20} {'Comparison with DNN':<30}")
            print("-"*80)
            
            for method, avg_loss in classic_methods_avg_losses.items():
                diff = dnn_avg_loss - avg_loss
                relative_diff = diff / avg_loss * 100 if avg_loss != 0 else float('inf')
                
                if diff < 0:
                    comparison = f"DNN better by {abs(diff):.6f} ({abs(relative_diff):.2f}%)"
                elif diff > 0:
                    comparison = f"DNN worse by {diff:.6f} ({relative_diff:.2f}%)"
                else:
                    comparison = "Equal performance"
                    
                print(f"{method:<20} {avg_loss:<20.6f} {comparison:<30}")
        
        print("\n" + "="*80)
        
    def _save_results(self) -> None:
        """Save simulation results to the output directory."""
        logger.info(f"Saving results to {self.output_dir}")
        # Placeholder for results saving implementation
        
    def run_scenario(self, scenario_type: str, values: List[Any]) -> Dict[str, Dict[str, Any]]:
        """
        Run a parametric scenario with multiple values.
        
        Args:
            scenario_type: Type of scenario (e.g., "SNR", "T", "M", "eta")
            values: List of values to test
            
        Returns:
            Dict mapping scenario values to their results
        """
        logger.info(f"Running {scenario_type} scenario with values: {values}")
        scenario_results = {}
        
        for value in values:
            logger.info(f"Running scenario with {scenario_type}={value}")
            
            # Create a modified configuration for this scenario
            from config.loader import apply_overrides
            modified_config = apply_overrides(
                self.config,
                [f"system_model.{scenario_type.lower()}={value}"]
            )
            
            # Create a new simulation with the modified config
            simulation = Simulation(
                config=modified_config,
                components=self.components,  # Reuse components where possible
                output_dir=self.output_dir / f"{scenario_type}_{value}"
            )
            
            # Run the simulation and store results
            result = simulation.run()
            scenario_results[value] = result
            
        self.results[scenario_type] = scenario_results
        return scenario_results

    def _create_trajectory_handler(self) -> None:
        """Create a trajectory data handler if not already present."""
        if self.trajectory_handler is None and self.config.trajectory.enabled:
            from .runners.data import TrajectoryDataHandler
            
            # Create trajectory handler
            logger.info("Creating trajectory data handler")
            self.trajectory_handler = TrajectoryDataHandler(
                system_model_params=self.system_model.params,
                config=self.config
            )
            
            # Store in components
            self.components["trajectory_handler"] = self.trajectory_handler
            
            logger.info("Trajectory data handler created successfully") 

    def _create_trajectory_dataset_for_testing(self, samples_size: int) -> Tuple[Any, Any]:
        """Create a trajectory dataset specifically for testing."""
        return self.trajectory_handler.create_dataset(
            samples_size=samples_size,
            trajectory_length=self.config.trajectory.trajectory_length,
            trajectory_type=self.config.trajectory.trajectory_type,
            save_dataset=False,  # Don't save test datasets
            dataset_path=Path("data/datasets").absolute()
        )
    
    def _create_standard_test_dataset(self, samples_size: int) -> Any:
        """Create a standard (non-trajectory) dataset for testing."""
        from DCD_MUSIC.src.signal_creation import Samples
        from DCD_MUSIC.src.data_handler import create_dataset
        
        samples_model = Samples(self.system_model.params)
        dataset, _ = create_dataset(
            samples_model=samples_model,
            samples_size=samples_size,
            save_datasets=False,  # Don't save test datasets
            datasets_path=Path("data/datasets").absolute(),
            true_doa=self.config.dataset.true_doa_test if hasattr(self.config.dataset, "true_doa_test") else self.config.dataset.true_doa_train,
            true_range=self.config.dataset.true_range_test if hasattr(self.config.dataset, "true_range_test") else self.config.dataset.true_range_train,
            phase="test"
        )
        return dataset 

    def _load_and_apply_weights(self, model_path: Path, device: torch.device) -> Tuple[bool, Optional[str]]:
        """Loads weights from a checkpoint file and applies them to self.model."""
        logger.info(f"Attempting to load model weights from: {model_path}")
        try:
            # First try without any specific parameters
            try:
                # Simple approach - let PyTorch handle it based on version
                state_dict = torch.load(model_path, map_location=device)
            except Exception as e:
                # If any error occurs, try the backward compatibility mode
                logger.warning(f"Standard loading failed, attempting with backward compatibility mode: {e}")
                try:
                    # weights_only=False can help with older PyTorch versions or complex saved states
                    state_dict = torch.load(model_path, map_location=device, weights_only=False)
                except Exception as e2:
                    # If both methods fail, raise a comprehensive error
                    raise RuntimeError(f"Failed to load checkpoint file with both standard and compatibility methods: {e}, then: {e2}")

            # Handle various checkpoint formats intelligently
            original_state_dict = state_dict # Keep a reference before modification
            if isinstance(state_dict, dict):
                # Common keys where model weights might be stored
                possible_keys = ['model_state_dict', 'state_dict', 'model', 'network', 'net_state_dict',
                               'net', 'weights', 'params', 'parameters']

                # Check if this is a checkpoint with nested weights or direct weights
                model_keys = self.model.state_dict().keys()
                # Check if *any* key from the model exists at the top level of the loaded dict
                is_direct_weights = any(k in state_dict for k in model_keys)

                if is_direct_weights:
                    logger.info("Checkpoint contains direct model weights - using as-is")
                else:
                    # Try to find weights in nested dictionary
                    found_key = None
                    for key in possible_keys:
                        if key in state_dict and isinstance(state_dict[key], dict):
                            nested_dict = state_dict[key]
                            # Check if the nested dict looks like model weights
                            # Check if *any* key from the model exists in the nested dict
                            if any(k in nested_dict for k in model_keys):
                                found_key = key
                                logger.info(f"Found model weights in checkpoint under '{key}' key")
                                state_dict = nested_dict
                                break

                    if not found_key:
                        logger.warning(f"Could not find model weights in checkpoint. Available keys: {list(state_dict.keys())}")
                        logger.warning("Will attempt to use checkpoint as-is - this may fail")
            else:
                 # Handle cases where torch.load returns something other than a dict
                 logger.warning(f"Loaded checkpoint is not a dictionary (type: {type(state_dict)}). Attempting to use as-is.")
                 # If it's not a dict, we probably can't load it directly anyway, but let the next step try.

            # Apply state dict to the model
            try:
                # Check for DataParallel/DistributedDataParallel wrapper
                # which adds 'module.' prefix to all keys
                if isinstance(state_dict, dict) and all(k.startswith('module.') for k in state_dict.keys()):
                    logger.info("Detected DataParallel wrapped model weights - removing 'module.' prefix")
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                incompatible_keys = self.model.load_state_dict(state_dict, strict=True)
                # If strict=True succeeds, incompatible_keys should be empty for both missing and unexpected
                logger.info("Model weights loaded successfully (strict=True)")
                self.model = self.model.to(device)
                self.trained_model = self.model # Mark as loaded/trained
                return True, "Model loaded successfully"

            except RuntimeError as e:
                # This often happens when model architectures don't match or state_dict is wrong
                logger.warning(f"Strict loading failed: {e}. Attempting partial loading (strict=False).")

                # Get model architecture info for debugging, using the potentially modified state_dict
                if isinstance(state_dict, dict):
                    model_keys_set = set(self.model.state_dict().keys())
                    state_dict_keys_set = set(state_dict.keys())
                    missing_keys = model_keys_set - state_dict_keys_set
                    unexpected_keys = state_dict_keys_set - model_keys_set
                    if missing_keys:
                        logger.warning(f"Missing keys in checkpoint: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
                else:
                    logger.warning("Cannot perform key comparison: loaded state_dict is not a dictionary.")


                # Try partial loading with strict=False as fallback
                try:
                    # We need to re-apply the 'module.' removal if it happened before strict=True failed
                    if isinstance(state_dict, dict) and all(k.startswith('module.') for k in state_dict.keys()):
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                    # Need to handle the case where state_dict wasn't a dict from the start
                    if not isinstance(state_dict, dict):
                         raise TypeError(f"Cannot load state_dict: Expected a dictionary, but got {type(state_dict)}")

                    incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
                    # Log specifics about partial load
                    if incompatible_keys.missing_keys:
                         logger.warning(f"Partial loading: Missing keys: {incompatible_keys.missing_keys}")
                    if incompatible_keys.unexpected_keys:
                         logger.warning(f"Partial loading: Unexpected keys: {incompatible_keys.unexpected_keys}")

                    # Check if *any* weights were actually loaded
                    if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                         logger.info("Partial loading (strict=False) completed successfully with no incompatible keys.")
                    elif len(incompatible_keys.missing_keys) < len(self.model.state_dict()):
                         logger.warning("Partial loading (strict=False) completed, but some keys were missing or unexpected.")
                    else:
                         # If all keys are missing, it's essentially a failure
                         raise RuntimeError("Partial loading failed: No matching keys found.")

                    self.model = self.model.to(device)
                    self.trained_model = self.model # Mark as loaded/trained
                    msg = f"Model partially loaded. Missing: {incompatible_keys.missing_keys}, Unexpected: {incompatible_keys.unexpected_keys}"
                    logger.info(msg)
                    # Returning True because *some* weights were loaded, but message indicates partial success
                    return True, msg
                except Exception as e2:
                    logger.error(f"Strict and partial loading both failed: {e2}")
                    return False, f"Failed to apply state_dict to model. Strict error: {e}. Partial error: {e2}"

        except Exception as load_err:
            logger.error(f"Failed to load or apply model weights from {model_path}: {load_err}")
            return False, f"Error during model loading: {load_err}" 

    def _save_model_state(self, model, model_type=None):
        """
        Save model state to a timestamped file in the checkpoints directory.
        
        Args:
            model: Model to save
            model_type: Type of model for filename (uses config if None)
            
        Returns:
            Path to saved model or None if save failed
        """
        if model is None:
            logger.warning("Cannot save model: model is None")
            return None
        
        model_save_dir = self.output_dir / "checkpoints"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine model type from config if not provided
        if model_type is None:
            model_type = self.config.model.type
        
        model_save_path = model_save_dir / f"saved_{model_type}_{timestamp}.pt"
        
        logger.info(f"Saving model to {model_save_path}")
        
        # Save only the state_dict, not the entire model
        try:
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved successfully to {model_save_path}")
            return model_save_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None

    def _run_online_learning(self) -> None:
        """
        Execute online learning on a single trajectory with drift detection.
        
        This method:
        1. Loads a pre-trained model
        2. Processes trajectory data window-by-window
        3. Evaluates performance on each window to detect drift
        4. Performs unsupervised online training when drift is detected
        5. Tracks loss/performance over time
        
        Returns:
            None, but stores results in self.results
        """
        logger.info("Starting online learning pipeline")
        
        # Validate that we have a model to use
        if self.trained_model is None and self.model is not None:
            logger.info("Using loaded model for online learning")
            self.trained_model = self.model
        elif self.trained_model is None:
            logger.error("No model available for online learning")
            self.results["online_learning_error"] = "No model available"
            return
        
        # Ensure model is in evaluation mode initially
        self.trained_model.eval()
        
        # Get online learning configuration parameters
        if not hasattr(self.config, 'online_learning'):
            logger.error("No online_learning configuration section found")
            self.results["online_learning_error"] = "Missing online_learning configuration"
            return
        
        window_size = getattr(self.config.online_learning, 'window_size', 10)
        stride = getattr(self.config.online_learning, 'stride', 5)
        loss_threshold = getattr(self.config.online_learning, 'loss_threshold', 0.5)
        max_iterations = getattr(self.config.online_learning, 'max_iterations', 10)
        
        # Create trajectory handler if needed
        if self.trajectory_handler is None:
            self._create_trajectory_handler()
        
        if self.trajectory_handler is None:
            logger.error("Failed to create trajectory handler")
            self.results["online_learning_error"] = "Failed to create trajectory handler"
            return
        
        try:
            # Create online learning dataset
            from simulation.runners.data import create_online_learning_dataset
            
            logger.info(f"Creating online learning dataset with window_size={window_size}, stride={stride}")
            online_dataset = create_online_learning_dataset(
                self.trajectory_handler,
                self.config,
                window_size=window_size,
                stride=stride
            )
            
            # Create dataloader
            dataloader = online_dataset.get_dataloader(batch_size=1, shuffle=False)
            logger.info(f"Created online learning dataloader with {len(dataloader)} windows")
            
            # Initialize results containers
            drift_detected_count = 0
            model_updated_count = 0
            window_losses = []
            window_update_flags = []
            
            # Initialize Kalman filter
            kf_Q, kf_R, kf_P0 = KalmanFilter1D.from_config(self.config)
            batch_kf = BatchKalmanFilter1D.from_config(
                self.config,
                batch_size=1,  # Single trajectory
                max_sources=self.system_model.params.M,
                device=device
            )
            
            # Initialize online trainer
            from simulation.runners.training import OnlineTrainer
            online_trainer = OnlineTrainer(
                model=self.trained_model,
                config=self.config,
                device=device
            )
            
            # Process each window
            with torch.no_grad():
                for window_idx, trajectory_window in enumerate(tqdm(dataloader, desc="Online Learning")):
                    # Unpack window data
                    time_series, sources_num, labels = trajectory_window
                    
                    # Calculate loss on window
                    window_loss = self._evaluate_window(time_series, sources_num, labels)
                    window_losses.append(window_loss)
                    
                    logger.info(f"Window {window_idx}: Loss = {window_loss:.6f}")
                    
                    # Check if loss exceeds threshold (drift detected)
                    if window_loss > loss_threshold:
                        logger.info(f"Drift detected in window {window_idx} (loss: {window_loss:.6f} > threshold: {loss_threshold:.6f})")
                        drift_detected_count += 1
                        
                        # Re-enable gradients for training
                        with torch.enable_grad():
                            # Perform online training
                            updated_model, updated_loss = online_trainer.train_on_window(
                                trajectory_window,
                                max_iterations=max_iterations
                            )
                            
                            # Update model if training improved performance
                            if updated_loss < window_loss:
                                logger.info(f"Model updated: Loss improved from {window_loss:.6f} to {updated_loss:.6f}")
                                self.trained_model = updated_model
                                model_updated_count += 1
                                window_update_flags.append(True)
                            else:
                                logger.info(f"Model update rejected: Loss did not improve ({updated_loss:.6f} >= {window_loss:.6f})")
                                window_update_flags.append(False)
                    else:
                        logger.info(f"No drift detected in window {window_idx} (loss: {window_loss:.6f} <= threshold: {loss_threshold:.6f})")
                        window_update_flags.append(False)
                
                # Save final model if it was updated
                if model_updated_count > 0:
                    model_save_path = self._save_model_state(
                        self.trained_model,
                        model_type=f"{self.config.model.type}_online_updated"
                    )
                    logger.info(f"Saved final online-updated model to {model_save_path}")
            
            # Store results
            self.results["online_learning"] = {
                "window_losses": window_losses,
                "window_updates": window_update_flags,
                "drift_detected_count": drift_detected_count,
                "model_updated_count": model_updated_count,
                "window_count": len(dataloader),
                "window_size": window_size,
                "stride": stride,
                "loss_threshold": loss_threshold
            }
            
            # Plot results
            self._plot_online_learning_results(window_losses, window_update_flags)
            
            logger.info(f"Online learning completed: {drift_detected_count} drifts detected, {model_updated_count} model updates")
            
        except Exception as e:
            logger.exception(f"Error during online learning: {e}")
            self.results["online_learning_error"] = str(e)

    def _evaluate_window(self, window_time_series, window_sources_num, window_labels):
        """
        Calculate loss on a window of trajectory data.
        
        Args:
            window_time_series: Time series data for window [batch, window_size, N, T]
            window_sources_num: Source counts for window [batch, window_size]
            window_labels: Labels for window (list of tensors)
            
        Returns:
            Average loss across window
        """
        # Assuming batch size of 1 for online learning
        time_series = window_time_series[0]
        sources_num = window_sources_num[0]
        labels = window_labels[0]
        
        window_size = time_series.shape[0]
        total_loss = 0.0
        
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near"
        
        # Use RMSPE loss for evaluation
        from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
        rmspe_criterion = RMSPELoss().to(device)
        
        # Process each step in window
        for step in range(window_size):
            # Extract data for this step
            step_data = time_series[step:step+1].to(device)  # Add batch dimension
            step_sources = sources_num[step].item()
            
            # Skip if no sources
            if step_sources <= 0:
                continue
            
            # Forward pass through model
            if not is_near_field:
                angles_pred, _, _ = self.trained_model(step_data, step_sources)
                
                # Get ground truth labels
                step_labels = torch.tensor(labels[step][:step_sources]).unsqueeze(0).to(device)
                
                # Calculate loss
                loss = rmspe_criterion(angles_pred, step_labels)
            else:
                # Near-field case
                angles_pred, ranges_pred, _, _ = self.trained_model(step_data, step_sources)
                
                # Get ground truth labels (assuming format matches)
                step_labels = torch.tensor(labels[step]).unsqueeze(0).to(device)
                angles_gt = step_labels[:, :step_sources]
                ranges_gt = step_labels[:, step_sources:step_sources*2]
                
                # Calculate loss (combined angle and range)
                angle_loss = rmspe_criterion(angles_pred, angles_gt)
                range_loss = rmspe_criterion(ranges_pred, ranges_gt)
                loss = angle_loss + range_loss
            
            total_loss += loss.item()
        
        # Calculate average loss
        if window_size > 0:
            return total_loss / window_size
        else:
            return float('inf')

    def _plot_online_learning_results(self, window_losses, window_updates):
        """
        Plot online learning results.
        
        Args:
            window_losses: List of loss values per window
            window_updates: List of boolean flags indicating model updates
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Create x-axis values
            x = np.arange(len(window_losses))
            
            # Plot losses
            plt.plot(x, window_losses, 'b-', label='Window Loss')
            
            # Highlight windows where model was updated
            update_indices = [i for i, updated in enumerate(window_updates) if updated]
            if update_indices:
                update_losses = [window_losses[i] for i in update_indices]
                plt.scatter(update_indices, update_losses, color='r', s=80, marker='o', label='Model Updated')
            
            # Add threshold line if configured
            if hasattr(self.config.online_learning, 'loss_threshold'):
                threshold = self.config.online_learning.loss_threshold
                plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold})')
            
            # Add labels and title
            plt.xlabel('Window Index')
            plt.ylabel('Loss')
            plt.title('Online Learning Results')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = self.output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"online_learning_results_{timestamp}.png"
            plt.savefig(plot_path)
            
            logger.info(f"Online learning results plot saved to {plot_path}")
            
        except ImportError:
            logger.warning("matplotlib not available for plotting online learning results")
        except Exception as e:
            logger.error(f"Error plotting online learning results: {e}") 
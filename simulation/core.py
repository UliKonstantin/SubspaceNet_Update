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

from config.schema import Config
from .runners.data import TrajectoryDataHandler
from .runners.training import Trainer, TrainingConfig, TrajectoryTrainer
from .runners.evaluation import Evaluator
from simulation.kalman_filter import KalmanFilter1D
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
            
            # Execute training pipeline if enabled
            if self.config.training.enabled:
                self._run_training_pipeline()
                
                # Check if training was successful
                if self.trained_model is None:
                    logger.error("Training pipeline failed, skipping evaluation")
                    return {"status": "error", "message": "Training pipeline failed"}
                    
            # Execute evaluation pipeline if enabled
            if self.config.simulation.evaluate_model:
                self._run_evaluation_pipeline()
                
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
        
    def _run_evaluation_pipeline_wrapper(self) -> None:
        """
        Execute the evaluation pipeline wrapper.
        
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
            
            # Configure Kalman Filter parameters
            kf_process_noise_std_dev = self.config.kalman_filter.process_noise_std_dev
            if kf_process_noise_std_dev is None:
                kf_process_noise_std_dev = self.config.trajectory.random_walk_std_dev
                logger.info(f"Using trajectory.random_walk_std_dev ({kf_process_noise_std_dev}) for KF process noise")
            
            # Calculate variances from standard deviations
            kf_Q = kf_process_noise_std_dev ** 2
            kf_R = self.config.kalman_filter.measurement_noise_std_dev ** 2
            kf_P0 = self.config.kalman_filter.initial_covariance
            
            logger.info(f"Kalman Filter parameters: Q={kf_Q}, R={kf_R}, P0={kf_P0}")
            
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
                    
                    # Process each trajectory separately
                    for traj_idx in range(batch_size):
                        # Extract single trajectory
                        single_trajectory = trajectories[traj_idx].to(device)
                        single_sources = sources_num[traj_idx].to(device)
                        
                        # Determine number of sources for this trajectory
                        num_sources = single_sources[0].item()
                        
                        # Get ground truth angles for this trajectory
                        traj_ground_truth = labels[traj_idx, :, :num_sources].cpu().numpy()
                        
                        # Get initial true angles for Kalman Filter initialization
                        initial_true_angles = traj_ground_truth[0, :]
                        
                        # Initialize Kalman filters - one per source
                        k_filters = [KalmanFilter1D(Q=kf_Q, R=kf_R, P0=kf_P0) for _ in range(num_sources)]
                        for s_idx in range(num_sources):
                            k_filters[s_idx].initialize_state(initial_true_angles[s_idx])
                        
                        # Initialize results for this trajectory
                        dnn_traj_preds = []
                        dnn_traj_kf_preds = []
                        
                        # Process each step in the trajectory
                        for step in range(trajectory_length):
                            # Step 1: Evaluate DNN model with Kalman filter
                            step_dnn_result = self._evaluate_dnn_model_kf_step(
                                single_trajectory[step],
                                single_sources[step],
                                labels[traj_idx, step, :num_sources],
                                k_filters,
                                rmspe_criterion,
                                num_sources,
                                is_near_field
                            )
                            
                            # Store DNN results
                            dnn_traj_preds.append(step_dnn_result["model_prediction"])
                            dnn_traj_kf_preds.append(step_dnn_result["kf_prediction"])
                            dnn_total_loss += step_dnn_result["loss"]
                            dnn_total_samples += 1
                            
                            # Step 2: Evaluate classic methods if enabled
                            if classic_methods:
                                # Get data for current step
                                step_data = single_trajectory[step]
                                step_sources = single_sources[step]
                                step_angles = labels[traj_idx, step, :num_sources].to(device)
                                
                                step_classic_results = self._evaluate_classic_methods_step(
                                    step_data,
                                    step_sources,
                                    step_angles,
                                    rmspe_criterion,
                                    classic_methods
                                )
                                
                                # Store classic method losses
                                for method in classic_methods:
                                    if method in step_classic_results:
                                        classic_methods_losses[method]["total_loss"] += step_classic_results[method]["loss"]
                                        classic_methods_losses[method]["total_samples"] += 1
                        
                        # Store trajectory results
                        dnn_trajectory_results.append({
                            'model_predictions': dnn_traj_preds,
                            'kf_predictions': dnn_traj_kf_preds,
                            'ground_truth': traj_ground_truth,
                            'sources': single_sources.cpu().numpy()
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

    def _evaluate_dnn_model_kf_step(
        self, 
        step_data: torch.Tensor, 
        step_sources: torch.Tensor,
        step_angles: torch.Tensor,
        k_filters: List[KalmanFilter1D],
        rmspe_criterion: RMSPELoss,
        num_sources: int,
        is_near_field: bool
    ) -> Dict[str, Any]:
        """
        Evaluate the DNN model for a single step and apply Kalman filtering.
        
        Args:
            step_data: Trajectory step data
            step_sources: Number of sources for this step
            step_angles: Ground truth angles for this step
            k_filters: List of Kalman filters (one per source)
            rmspe_criterion: Loss criterion for evaluation
            num_sources: Number of sources in this step
            is_near_field: Whether the model is for near-field estimation
            
        Returns:
            Dictionary containing model prediction, KF prediction, and loss
        """
        result = {}
        
        # Add batch dimension for model
        step_data = step_data.unsqueeze(0)
        step_sources = step_sources.unsqueeze(0)
        step_angles = step_angles.unsqueeze(0).to(device)
        
        # Perform Kalman Filter prediction for each source
        kf_step_predictions = np.zeros(num_sources)
        for s_idx in range(num_sources):
            kf_step_predictions[s_idx] = k_filters[s_idx].predict()
        
        # Store KF predictions before update
        result["kf_prediction"] = kf_step_predictions.copy()
        
        # Model forward pass
        if is_near_field:
            angles_pred, ranges_pred, _ = self.trained_model(step_data, step_sources)
            result["ranges_prediction"] = ranges_pred.cpu().numpy()
        else:
            angles_pred, _, _ = self.trained_model(step_data, step_sources)
        
        # Store model prediction
        result["model_prediction"] = angles_pred.cpu().numpy()
        
        # Update Kalman Filters with model predictions
        model_angles_step = angles_pred.squeeze().cpu().numpy()
        for s_idx in range(num_sources):
            # Get the measurement for this source
            measurement = model_angles_step[s_idx] if s_idx < len(model_angles_step) else model_angles_step
            # Update KF with model prediction
            k_filters[s_idx].update(measurement)
        
        # Calculate loss using RMSPELoss
        step_loss = rmspe_criterion(angles_pred, step_angles)
        result["loss"] = step_loss.item()
        
        return result

    def _evaluate_classic_methods_step(
        self,
        step_data: torch.Tensor,
        step_sources: torch.Tensor,
        step_angles: torch.Tensor,
        rmspe_criterion: RMSPELoss,
        methods: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate classic subspace methods for a single step.
        
        Args:
            step_data: Trajectory step data
            step_sources: Number of sources for this step
            step_angles: Ground truth angles for this step
            rmspe_criterion: Loss criterion for evaluation
            methods: List of classic subspace methods to evaluate
            
        Returns:
            Dictionary mapping method names to their results (prediction and loss)
        """
        results = {}
        
        try:
            # Get a copy of the existing system model params
            system_model_params = self.system_model.params
            
            # Store the original M value to restore later
            original_M = system_model_params.M
            
            # Temporarily update the number of sources
            num_sources = int(step_sources.item())
            system_model_params.M = num_sources
            
            # Format the data as expected by test_step
            # The sample_covariance function expects shape [batch_size, sensor_number, samples_number]
            # Our step_data shape is likely [T, N] where T=samples and N=sensors
            # We need to transpose it to [N, T] and then add a batch dimension to get [1, N, T]
            
            # Log the original shape for debugging
            logger.debug(f"Original step_data shape: {step_data.shape}")
            
            # Transpose and add batch dimension
            # Going from [T, N] to [1, N, T]
            if step_data.dim() == 2:
                # If step_data is [T, N], transpose to [N, T] and add batch dim
                processed_data = step_data.transpose(0, 1).unsqueeze(0)
            elif step_data.dim() == 3:
                # If already [B, T, N], transpose to [B, N, T]
                processed_data = step_data.transpose(1, 2)
            else:
                # Handle unexpected shapes
                logger.error(f"Unexpected step_data shape: {step_data.shape}")
                return results
                
            logger.debug(f"Processed data shape: {processed_data.shape}")
            
            # Format the data as expected by test_step
            # test_step expects a tuple of (x, sources_num, label, masks)
            batch_data = (
                processed_data,  # Shape should be [1, N, T]
                step_sources.unsqueeze(0),  # Add batch dimension to sources [1]
                step_angles.unsqueeze(0),  # Add batch dimension to angles [1, M]
                torch.ones_like(step_angles).unsqueeze(0)  # Add masks (all ones) [1, M]
            )
            
            for method_name in methods:
                try:
                    # This creates a method instance
                    classic_method = get_model_based_method(method_name, system_model_params)
                    
                    if classic_method is not None:
                        # Call test_step to get the loss and accuracy
                        loss, accuracy, test_length = classic_method.test_step(batch_data, 0)  # 0 is the batch index
                        
                        # Store only loss and accuracy (no predictions)
                        results[method_name] = {
                            "loss": loss,
                            "accuracy": accuracy
                        }
                    else:
                        logger.warning(f"Failed to initialize classic method: {method_name}")
                        logger.warning(f"Available methods may include: MUSIC, Root-MUSIC, ESPRIT")
                except Exception as method_error:
                    logger.error(f"Error with method {method_name}: {method_error}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Restore the original M value
            system_model_params.M = original_M
            
        except Exception as e:
            logger.error(f"Error evaluating classic methods: {e}")
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

    def _run_evaluation_pipeline(self) -> None:
        """
        Execute the evaluation pipeline.
        
        This is a wrapper method that calls _run_evaluation_pipeline_wrapper
        to maintain backward compatibility.
        """
        return self._run_evaluation_pipeline_wrapper() 
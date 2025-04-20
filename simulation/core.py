"""
Core simulation components and main controller.

This module contains the main Simulation class that orchestrates
the simulation process from data handling to evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from config.schema import Config
from .runners.data import TrajectoryDataHandler
from .runners.training import Trainer, TrainingConfig, TrajectoryTrainer
from .runners.evaluation import Evaluator

logger = logging.getLogger(__name__)

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
        
    def _run_evaluation_pipeline(self) -> None:
        """
        Execute the evaluation pipeline.
        
        Evaluates the model using trajectories for testing:
        1. Uses the test dataloader with trajectory data
        2. Loops through each trajectory and evaluates performance
        3. Applies Kalman filtering for tracking predictions
        4. Calculates and stores evaluation metrics
        """
        logger.info("Starting evaluation pipeline")
        
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
            logger.info("Evaluating model on trajectory test data")
            
            # Import necessary libraries
            import torch
            import numpy as np
            from tqdm import tqdm
            from simulation.kalman_filter import KalmanFilter1D
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Check if we're dealing with near-field model
            is_near_field = hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near"
            if is_near_field:
                error_msg = "Near-field option is not available in the current evaluation pipeline"
                logger.error(error_msg)
                self.results["evaluation_error"] = error_msg
                return
            
            # Configure Kalman Filter parameters
            # Get process noise from config (use random_walk_std_dev if process_noise_std_dev is not set)
            kf_process_noise_std_dev = self.config.kalman_filter.process_noise_std_dev
            if kf_process_noise_std_dev is None:
                kf_process_noise_std_dev = self.config.trajectory.random_walk_std_dev
                logger.info(f"Using trajectory.random_walk_std_dev ({kf_process_noise_std_dev}) for KF process noise")
            
            # Calculate variances from standard deviations
            kf_Q = kf_process_noise_std_dev ** 2
            kf_R = self.config.kalman_filter.measurement_noise_std_dev ** 2
            kf_P0 = self.config.kalman_filter.initial_covariance
            
            logger.info(f"Kalman Filter parameters: Q={kf_Q}, R={kf_R}, P0={kf_P0}")
            
            # Results container
            trajectory_results = []
            test_metrics = {}
            total_loss = 0.0
            total_samples = 0
            
            # Trajectory-level evaluation with Kalman filtering
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
                        
                        # Initialize results for this trajectory
                        traj_preds = []
                        traj_ground_truth = []
                        traj_kf_preds = []  # Kalman filter predictions
                        
                        # Determine number of sources for this trajectory
                        # For simplicity, we assume constant sources per trajectory
                        num_sources = single_sources[0].item()
                        
                        # Get initial true angles for this trajectory
                        initial_true_angles = labels[traj_idx, 0, :num_sources].cpu().numpy()
                        
                        # Initialize Kalman filters - one per source
                        k_filters = [KalmanFilter1D(Q=kf_Q, R=kf_R, P0=kf_P0) for _ in range(num_sources)]
                        for s_idx in range(num_sources):
                            k_filters[s_idx].initialize_state(initial_true_angles[s_idx])
                        
                        # Process each step in the trajectory
                        for step in range(trajectory_length):
                            # Get data for current step
                            step_data = single_trajectory[step].unsqueeze(0)  # Add batch dimension
                            step_sources = single_sources[step].unsqueeze(0)
                            
                            # Extract angles (for far-field)
                            step_angles = labels[traj_idx, step, :num_sources].unsqueeze(0).to(device)
                            
                            # Perform Kalman Filter prediction for each source
                            kf_step_predictions = np.zeros(num_sources)
                            for s_idx in range(num_sources):
                                kf_step_predictions[s_idx] = k_filters[s_idx].predict()
                            
                            # Store KF predictions before update
                            traj_kf_preds.append(kf_step_predictions.copy())
                            
                            # Model forward pass
                            angles_pred, _, _ = self.trained_model(step_data, step_sources)
                            
                            # Store model predictions and ground truth
                            traj_preds.append(angles_pred.cpu().numpy())
                            traj_ground_truth.append(step_angles.cpu().numpy())
                            
                            # Update Kalman Filters with model predictions
                            model_angles_step = angles_pred.squeeze().cpu().numpy()
                            for s_idx in range(num_sources):
                                # Get the measurement for this source
                                measurement = model_angles_step[s_idx] if num_sources > 1 else model_angles_step
                                # Update KF with model prediction
                                k_filters[s_idx].update(measurement)
                            
                            # Calculate loss for metrics
                            step_loss = torch.nn.functional.mse_loss(angles_pred, step_angles)
                            
                            # Accumulate loss
                            total_loss += step_loss.item()
                            total_samples += 1
                        
                        # Calculate metrics for this trajectory
                        # TODO: Implement trajectory-specific metrics
                        # (e.g., average prediction error, tracking stability)
                        
                        # Store trajectory results
                        trajectory_results.append({
                            'predictions': traj_preds,
                            'ground_truth': traj_ground_truth,
                            'kf_predictions': traj_kf_preds,
                            'sources': single_sources.cpu().numpy()
                        })
                
                # Calculate overall metrics
                test_loss = total_loss / total_samples
                test_metrics['loss'] = test_loss
                
                # TODO: Calculate additional metrics using trajectory_results
                # (e.g., average trajectory tracking error, convergence time)
            
            # Store results
            self.results["test_loss"] = test_loss
            self.results["test_metrics"] = test_metrics
            self.results["trajectory_results"] = trajectory_results
            
            # Log results
            logger.info(f"Test loss: {test_loss:.6f}")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"Test {metric_name}: {metric_value:.6f}")
            
            logger.info(f"Evaluated {len(trajectory_results)} trajectories")
            logger.info("Kalman filter applied to trajectory predictions")
                
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            self.results["evaluation_error"] = str(e)
    
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
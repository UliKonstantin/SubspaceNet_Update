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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
import copy
import yaml
from itertools import permutations
import itertools

from config.schema import Config
from .runners.data import TrajectoryDataHandler
from .runners.training import Trainer, TrainingConfig, TrajectoryTrainer
from .runners.evaluation import Evaluator
from .runners.Online_learning import OnlineLearning
from simulation.kalman_filter import KalmanFilter1D, BatchKalmanFilter1D, BatchExtendedKalmanFilter1D
from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
from DCD_MUSIC.src.signal_creation import Samples
from DCD_MUSIC.src.evaluation import get_model_based_method, evaluate_model_based
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D

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
        Run a single simulation with all components (legacy method).
        
        This method runs all components in sequence: training, evaluation, and online learning.
        For more focused execution, use run_training, run_evaluation, or execute_online_learning.
        
        Returns:
            Dict containing simulation results.
        """
        logger.info("Starting complete simulation (training, evaluation, and online learning)")
        
        # Run training first
        training_results = self.run_training()
        
        # Check if training was successful
        if training_results.get("status") == "error":
            return training_results
            
        # Run evaluation if enabled
        if self.config.simulation.evaluate_model:
            evaluation_results = self.run_evaluation()
            # Merge results
            for key, value in evaluation_results.items():
                if key != "status":  # Don't overwrite status from training
                    training_results[key] = value
                    
        # Run online learning if enabled
        if hasattr(self.config, 'online_learning') and getattr(self.config.online_learning, 'enabled', False):
            online_results = self.execute_online_learning()
            # Merge results
            for key, value in online_results.items():
                if key != "status":  # Don't overwrite status from training
                    training_results[key] = value
                    
        return training_results
        
    def run_training(self) -> Dict[str, Any]:
        """
        Run the training pipeline.
        
        This method focuses on dataset creation and model training.
        
        Returns:
            Dict containing training results.
        """
        logger.info("Starting training pipeline")
        
        try:
            # Execute data pipeline for training scenario only
            self._run_data_pipeline(scenario="training")
            
            # Check if data pipeline was successful
            if self.train_dataloader is None:
                logger.error("Data pipeline failed, skipping training")
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
                    logger.error("Training pipeline failed")
                    return {"status": "error", "message": "Training pipeline failed"}
            elif not self.config.simulation.train_model:
                logger.info("Skipping training (simulation.train_model=False)")
            elif not self.config.training.enabled:
                logger.info("Skipping training (training.enabled=False)")
            
            # Save model if configured
            if self.config.simulation.save_model and self.trained_model is not None:
                self._save_model_state(self.trained_model)
            
            # Save results
            self._save_results()
            
            return {"status": "success", "trained_model": True if self.trained_model is not None else False}
            
        except Exception as e:
            logger.exception(f"Error running training: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the evaluation pipeline.
        
        This method focuses on model evaluation against test data.
        
        Returns:
            Dict containing evaluation results.
        """
        logger.info("Starting evaluation pipeline")
        
        try:
            # Create evaluation dataset only
            self._run_data_pipeline(scenario="evaluation")
            
            # Check if data pipeline was successful
            if self.test_dataloader is None:
                logger.error("Data pipeline failed, cannot create test dataset")
                return {"status": "error", "message": "Failed to create test dataset"}
            
            # Load model from path if configured
            if self.config.simulation.load_model:
                if hasattr(self.config.simulation, 'model_path') and self.config.simulation.model_path:
                    model_path = Path(self.config.simulation.model_path)
                    logger.info(f"Loading model from path: {model_path}")
                    success, message = self._load_and_apply_weights(model_path, device)
                    if not success:
                        logger.error(f"Failed to load model: {message}")
                        return {"status": "error", "message": message}
                else:
                    logger.error("Model loading requested but no model_path provided in config")
                    return {"status": "error", "message": "No model_path provided"}
            
            # Load model if not already loaded
            if self.trained_model is None:
                if self.model is not None:
                    logger.info("Using non-trained model for evaluation")
                    self.trained_model = self.model
                else:
                    logger.error("No model available for evaluation")
                    return {"status": "error", "message": "No model available for evaluation"}
            
            # Run the evaluation
            self._run_evaluation_pipeline()
            
            # Save results
            self._save_results()
            
            return {"status": "success", "evaluation_results": self.results}
            
        except Exception as e:
            logger.exception(f"Error running evaluation: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}
    
    def execute_online_learning(self) -> Dict[str, Any]:
        """
        Execute the online learning pipeline.
        
        This method focuses on online model adaptation with streaming data.
        
        Returns:
            Dict containing online learning results.
        """
        logger.info("Starting online learning pipeline")
        
        try:
            # Load model from path if configured
            if self.config.simulation.load_model:
                if hasattr(self.config.simulation, 'model_path') and self.config.simulation.model_path:
                    model_path = Path(self.config.simulation.model_path)
                    logger.info(f"Loading model from path: {model_path}")
                    success, message = self._load_and_apply_weights(model_path, device)
                    if not success:
                        logger.error(f"Failed to load model: {message}")
                        return {"status": "error", "message": message}
                else:
                    logger.error("Model loading requested but no model_path provided in config")
                    return {"status": "error", "message": "No model_path provided"}
            
            # Check if model is available for online learning
            if self.trained_model is None:
                if self.model is not None:
                    logger.info("Using non-trained model for online learning")
                    self.trained_model = self.model
                else:
                    logger.error("No model available for online learning")
                    return {"status": "error", "message": "No model available for online learning"}
            
            # Create OnlineLearning handler and run the pipeline
            online_learning_handler = OnlineLearning(
                config=self.config,
                system_model=self.system_model,
                trained_model=self.trained_model,
                output_dir=self.output_dir,
                results=self.results
            )
            
            # Run online learning pipeline
            result = online_learning_handler.run_online_learning()
            
            # Save results
            self._save_results()
            
            return result
            
        except Exception as e:
            logger.exception(f"Error running online learning: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

    def _run_data_pipeline(self, scenario: str = "training") -> None:
        """
        Execute the data preparation pipeline.
        
        Creates or loads datasets based on configuration and
        sets up dataloaders for training, validation, and testing.
        
        This method handles all dataset creation for the simulation, enforcing a separation
        of concerns from the factory system. All components that need access to datasets
        (like trainers) will be updated with the datasets created here.
        
        Args:
            scenario: The scenario to prepare data for. One of:
                - "training": Creates training, validation, and test datasets
                - "evaluation": Creates only test dataset
                - "online_learning": Creates only online learning dataset
        """
        logger.info(f"Starting data pipeline for scenario: {scenario}")
        
        # For "evaluation" scenario, we only need the test dataset
        if scenario == "evaluation":
            self._prepare_test_dataset()
            return
            
        # For "online_learning" scenario, we only need the online learning dataset
        if scenario == "online_learning":
            self._prepare_online_learning_dataset()
            return
            
        # For "training" scenario, we need training, validation, and test datasets
        if scenario == "training":
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
                # Always create/load datasets here in the data pipeline, not in the factory
                if self.config.dataset.create_data:
                    logger.info("Creating new dataset")
                    dataset = self._create_standard_dataset()
                else:
                    logger.info("Loading existing dataset")
                    dataset = self._load_standard_dataset()
            
            if dataset is None:
                logger.error("Failed to create or load dataset")
                return
            
            self.dataset = dataset
            # Store dataset in components
            self.components["dataset"] = dataset
            
            # Update trainer with dataset if trainer exists
            if "trainer" in self.components and self.components["trainer"] is not None:
                logger.info("Updating trainer with created dataset")
                self.components["trainer"].dataset = dataset
            
            # Create dataloaders for training and validation
            logger.info("Creating dataloaders with batch size: %d", self.config.training.batch_size)
            
            # Get split proportions [test, validation, train]
            splits = getattr(self.config.dataset, "test_validation_train_split", [0.1, 0.1, 0.8])
            # Normalize to ensure they sum to 1.0
            total = sum(splits)
            if abs(total - 1.0) > 1e-6:
                splits = [p / total for p in splits]
            
            test_prop, val_prop, train_prop = splits
            logger.info(f"Using split: test={test_prop:.2f}, val={val_prop:.2f}, train={train_prop:.2f}")
            
            # Calculate validation split relative to train+val
            val_split = val_prop / (train_prop + val_prop) if (train_prop + val_prop) > 0 else 0
            
            # Create train and validation dataloaders
            self.train_dataloader, self.valid_dataloader = dataset.get_dataloaders(
                batch_size=self.config.training.batch_size,
                validation_split=val_split
            )
            
            logger.info(f"Created dataloaders: train={len(self.train_dataloader)}, val={len(self.valid_dataloader)}")
            
            # Store dataloaders in components
            self.components["train_dataloader"] = self.train_dataloader
            self.components["valid_dataloader"] = self.valid_dataloader
            
            # Create test dataset for training scenario
            self._prepare_test_dataset()
            return
            
        # If we get here, an invalid scenario was specified
        logger.error(f"Invalid scenario: {scenario}")
        raise ValueError(f"Invalid scenario: {scenario}")
    
    def _prepare_test_dataset(self) -> None:
        """Create and set up the test dataset and dataloader."""
        # Calculate test dataset size
        splits = getattr(self.config.dataset, "test_validation_train_split", [0.1, 0.1, 0.8])
        # Normalize
        total = sum(splits)
        if abs(total - 1.0) > 1e-6:
            splits = [p / total for p in splits]
        
        test_prop = splits[0]
        test_samples_size = max(1, int(self.config.dataset.samples_size * test_prop))
        logger.info(f"Creating test dataset: {test_samples_size} samples ({test_prop:.0%})")
        
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
    
    def _prepare_online_learning_dataset(self) -> None:
        """Create and set up the online learning dataset and dataloader."""
        # Create online learning dataset and dataloader if enabled
        if hasattr(self.config, 'online_learning') and getattr(self.config.online_learning, 'enabled', False):
            logger.info("Online learning is enabled, creating on-demand dataset and dataloader.")
            if not self.config.trajectory.enabled:
                logger.warning("Online learning typically relies on trajectory mode for continuous data generation. Ensure config is appropriate if trajectory.enabled is False.")
            
            # Ensure system_model and its params are available, as they are crucial for the generator
            if self.system_model is None or not hasattr(self.system_model, 'params') or self.system_model.params is None:
                logger.error("SystemModel or its params not available for online learning dataset creation. Online learning will be skipped.")
                self.online_learning_dataloader = None
            else:
                # Use the refactored factory from simulation.runners.data
                from simulation.runners.data import create_online_learning_dataset 
                
                online_config = self.config.online_learning
                window_size = getattr(online_config, 'window_size', 10)
                stride = getattr(online_config, 'stride', 5) 
                
                try:
                    # Key change: Pass the shared self.system_model.params object.
                    # This allows dynamic eta updates within the generator to be reflected.
                    online_dataset = create_online_learning_dataset(
                        system_model_params=self.system_model.params, # Shared instance
                        config=self.config, # Pass full config for trajectory_length etc.
                        window_size=window_size,
                        stride=stride
                    )
                    # Batch size is 1 for online learning as we process window by window
                    self.online_learning_dataloader = online_dataset.get_dataloader(batch_size=1, shuffle=False) 
                    logger.info(f"Created on-demand online learning dataloader for {len(self.online_learning_dataloader)} windows.")
                    self.components["online_learning_dataloader"] = self.online_learning_dataloader
                except ValueError as e:
                    logger.error(f"Error creating online learning dataset: {e}. Online learning may not function.")
                    self.online_learning_dataloader = None
                except Exception as e:
                    logger.exception(f"Unexpected error during online learning dataset creation: {e}")
                    self.online_learning_dataloader = None
        else:
            self.online_learning_dataloader = None # Ensure it's defined even if not used
            logger.info("Online learning is not enabled. Skipping online learning dataset creation.")
    
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
        # Directly create the dataset using DCD_MUSIC components
        from DCD_MUSIC.src.signal_creation import Samples
        from DCD_MUSIC.src.data_handler import create_dataset
        
        try:
            samples_model = Samples(self.system_model.params)
            # Store samples_model for potential later use
            self.components["samples_model"] = samples_model
            
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
        except Exception as e:
            logger.error(f"Failed to create standard dataset: {e}")
            return None
    
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
            
            # Check filter type configuration
            filter_type = getattr(self.config.kalman_filter, "filter_type", "standard").lower()
            logger.info(f"Using Kalman filter type: {filter_type}")
            
            # Get Kalman filter parameters from config (for backward compatibility)
            if filter_type == "standard":
                kf_Q, kf_R, kf_P0 = KalmanFilter1D.from_config(self.config)
            else:
                # For extended filter, we'll get parameters during batch filter creation
                kf_Q, kf_R, kf_P0 = None, None, None
            
            # Instantiate RMSPELoss criterion
            rmspe_criterion = RMSPELoss().to(device)
            
            # Result containers
            dnn_trajectory_results = []  # Store DNN+KF results
            
            # Overall metrics
            dnn_total_loss = 0.0
            ekf_total_loss = 0.0  # Track EKF loss separately
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
            
            # Prepare evaluator for classic methods
            evaluator = Evaluator(self.config, model=self.trained_model, system_model=self.system_model, output_dir=self.output_dir)
            
            # Trajectory-level evaluation with both DNN+KF and classic methods
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Evaluating trajectories")):
                    # Get batch data
                    trajectories, sources_num, labels = batch_data
                    batch_size, trajectory_length = trajectories.shape[0], trajectories.shape[1]
                    
                    # Initialize storage for batch results
                    batch_dnn_preds = [[] for _ in range(batch_size)]
                    batch_dnn_kf_preds = [[] for _ in range(batch_size)]
                    batch_kf_covariances = [[] for _ in range(batch_size)]  # Initialize covariance tracking
                    
                    # Find maximum number of sources in batch for efficient batch processing
                    max_sources = torch.max(sources_num).item()
                    
                    # Initialize the appropriate batch Kalman filter based on filter type
                    if filter_type == "extended":
                        logger.info(f"Creating BatchExtendedKalmanFilter1D for batch_size={batch_size}, max_sources={max_sources}")
                        batch_kf = BatchExtendedKalmanFilter1D.from_config(
                            self.config,
                            batch_size=batch_size,
                            max_sources=max_sources,
                            device=device,
                            trajectory_type=self.config.trajectory.trajectory_type
                        )
                    else:
                        logger.info(f"Creating BatchKalmanFilter1D for batch_size={batch_size}, max_sources={max_sources}")
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
                        model_preds, kf_preds, step_loss, kf_loss, step_covariances = self._evaluate_dnn_model_kf_step_batch(
                            step_data, step_sources, labels[:, step, :max_sources], step_mask, batch_kf, 
                            rmspe_criterion, is_near_field
                        )
                        
                        # Accumulate loss and store predictions
                        dnn_total_loss += step_loss
                        ekf_total_loss += kf_loss  # Accumulate EKF loss
                        dnn_total_samples += batch_size
                        
                        # Store predictions for this time step
                        for i in range(batch_size):
                            batch_dnn_preds[i].append(model_preds[i])
                            batch_dnn_kf_preds[i].append(kf_preds[i])
                            # Store covariances
                            batch_kf_covariances[i].append(step_covariances[i])
                        
                        # Evaluate classic methods if enabled
                        if classic_methods:
                            classic_results = evaluator._evaluate_classic_methods_step_batch(
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
                            'kf_covariances': batch_kf_covariances[i],  # Add KF covariances
                            'ground_truth': gt_trajectory,
                            'sources': num_sources_array
                        })
                
                # Log evaluation results
                self._log_evaluation_results(
                    dnn_total_loss, 
                    ekf_total_loss,  # Pass EKF total loss
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
        #TODO: replace with batch processing
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

            # Calculate loss for this trajectory and get best permutation
            with torch.no_grad():
                truth = step_angles[i, :num_sources_item].unsqueeze(0)
                angles_pred_single = angles_pred_single.view(1, -1)[:, :num_sources_item]
                loss, best_perm = rmspe_criterion(angles_pred_single, truth, return_best_perm=True)
                total_loss += loss.item()
                
                # Apply best permutation to predictions before appending
                angles_pred_optimal = angles_pred_single.squeeze(0)[best_perm.squeeze(0)]
                batch_angles_pred_list.append(angles_pred_optimal)

        # Combine predictions into a batch tensor for KF update
        # Need padding if source counts vary
        padded_angles_pred = torch.zeros((batch_size, max_sources), device=device)
        for i, pred in enumerate(batch_angles_pred_list):
            num_sources = pred.shape[0]
            if num_sources > 0:
                padded_angles_pred[i, :num_sources] = pred

        # Apply the Kalman filter to the predicted angles (using the padded batch tensor)
        # Get predictions before update (current state)
        kf_predictions_before_update = batch_kf.predict().cpu().numpy()

        # Then update with new measurements (model predictions) and get updated states
        kf_predictions_after_update, kf_covariances = batch_kf.update(padded_angles_pred, step_mask)
        kf_predictions_after_update = kf_predictions_after_update.cpu().numpy()
        kf_covariances = kf_covariances.cpu().numpy()

        # Convert predictions to list of numpy arrays with proper dimensions
        model_predictions_list = []
        kf_predictions_list = []
        kf_rmspe_list = []  # List to store RMSPE between KF update and truth
        kf_covariance_list = []  # List to store error covariances

        for i in range(batch_size):
            num_sources = step_sources[i].item()
            if num_sources > 0:
                model_predictions_list.append(padded_angles_pred[i, :num_sources].cpu().numpy())
                kf_predictions_list.append(kf_predictions_after_update[i, :num_sources])
                kf_covariance_list.append(kf_covariances[i, :num_sources])
                
                # Calculate RMSPE between KF update and truth
                with torch.no_grad():
                    kf_pred_tensor = torch.tensor(kf_predictions_after_update[i, :num_sources], device=device).unsqueeze(0)
                    truth_tensor = step_angles[i, :num_sources].unsqueeze(0)
                    kf_rmspe = rmspe_criterion(kf_pred_tensor, truth_tensor).item()
                    kf_rmspe_list.append(kf_rmspe)
            else:
                model_predictions_list.append(np.array([]))
                kf_predictions_list.append(np.array([]))
                kf_covariance_list.append(np.array([]))
                kf_rmspe_list.append(0.0)  # No sources, so RMSPE is 0

        return model_predictions_list, kf_predictions_list, total_loss, np.sum(kf_rmspe_list), kf_covariance_list

    def _log_evaluation_results(
        self,
        dnn_total_loss: float,
        ekf_total_loss: float,
        dnn_total_samples: int,
        classic_methods_losses: Dict[str, Dict[str, float]],
        dnn_trajectory_results: List[Dict[str, Any]],
        classic_trajectory_results: List[Dict[str, Any]]
    ) -> None:
        """
        Log the evaluation results for DNN, EKF, and classic methods.
        
        Args:
            dnn_total_loss: Accumulated loss for DNN model
            ekf_total_loss: Accumulated loss for EKF model
            dnn_total_samples: Total samples evaluated for DNN
            classic_methods_losses: Dictionary of losses for classic methods
            dnn_trajectory_results: List of trajectory results for DNN
            classic_trajectory_results: List of trajectory results for classic methods
        """
        # Calculate average losses for DNN and EKF models
        dnn_avg_loss = dnn_total_loss / max(dnn_total_samples, 1)
        ekf_avg_loss = ekf_total_loss / max(dnn_total_samples, 1)
        dnn_avg_loss_in_degrees = dnn_avg_loss * 180 / np.pi
        ekf_avg_loss_in_degrees = ekf_avg_loss * 180 / np.pi
        
        # Log DNN and EKF results
        logger.info(f"DNN Model - Average loss: {dnn_avg_loss:.6f} in degrees: {dnn_avg_loss_in_degrees:.6f}")
        logger.info(f"EKF Model - Average loss: {ekf_avg_loss:.6f} in degrees: {ekf_avg_loss_in_degrees:.6f}")
        
        # Calculate and log average losses for classic methods
        classic_methods_avg_losses = {}
        classic_methods_avg_losses_in_degrees = {}
        for method, loss_data in classic_methods_losses.items():
            if loss_data["total_samples"] > 0:
                avg_loss = loss_data["total_loss"] / loss_data["total_samples"]
                classic_methods_avg_losses[method] = avg_loss
                classic_methods_avg_losses_in_degrees[method] = avg_loss * 180 / np.pi
                logger.info(f"Classic Method {method} - Average loss: {avg_loss:.6f} in degrees: {classic_methods_avg_losses_in_degrees[method]:.6f}")
        
        # Store metrics in results
        self.results["dnn_test_loss"] = dnn_avg_loss
        self.results["ekf_test_loss"] = ekf_avg_loss  # Store EKF loss
        self.results["classic_methods_test_losses"] = classic_methods_avg_losses
        
        # Compare DNN vs EKF
        dnn_ekf_diff = dnn_avg_loss - ekf_avg_loss
        dnn_ekf_relative_diff = dnn_ekf_diff / ekf_avg_loss * 100 if ekf_avg_loss != 0 else float('inf')
        
        logger.info("DNN vs EKF Comparison:")
        if dnn_ekf_diff < 0:
            logger.info(f"DNN outperforms EKF by {abs(dnn_ekf_diff):.6f} ({abs(dnn_ekf_relative_diff):.2f}%) in degrees: {abs(dnn_ekf_diff * 180 / np.pi):.6f}")
        else:
            logger.info(f"EKF outperforms DNN by {dnn_ekf_diff:.6f} ({dnn_ekf_relative_diff:.2f}%) in degrees: {dnn_ekf_diff * 180 / np.pi:.6f}")
        
        # Compare DNN vs classic methods if both are available
        if classic_methods_avg_losses:
            logger.info("DNN vs Classic Methods Comparison:")
            for method, avg_loss in classic_methods_avg_losses.items():
                diff = dnn_avg_loss - avg_loss
                relative_diff = diff / avg_loss * 100 if avg_loss != 0 else float('inf')
                
                if diff < 0:
                    logger.info(f"DNN outperforms {method} by {abs(diff):.6f} ({abs(relative_diff):.2f}%) in degrees: {abs(diff * 180 / np.pi):.6f}")
                else:
                    logger.info(f"{method} outperforms DNN by {diff:.6f} ({relative_diff:.2f}%) in degrees: {diff * 180 / np.pi:.6f}")
            
            # Compare EKF vs classic methods
            logger.info("EKF vs Classic Methods Comparison:")
            for method, avg_loss in classic_methods_avg_losses.items():
                diff = ekf_avg_loss - avg_loss
                relative_diff = diff / avg_loss * 100 if avg_loss != 0 else float('inf')
                
                if diff < 0:
                    logger.info(f"EKF outperforms {method} by {abs(diff):.6f} ({abs(relative_diff):.2f}%) in degrees: {abs(diff * 180 / np.pi):.6f}")
                else:
                    logger.info(f"{method} outperforms EKF by {diff:.6f} ({relative_diff:.2f}%) in degrees: {diff * 180 / np.pi:.6f}")
        
        # Log total number of trajectories evaluated
        logger.info(f"Evaluated {len(dnn_trajectory_results)} trajectories with DNN model")
        if classic_trajectory_results:
            logger.info(f"Evaluated {len(classic_trajectory_results)} trajectories with classic methods")
            
        # Add print statements for detailed evaluation results
        print("\n" + "="*80)
        print(f"{'EVALUATION RESULTS':^80}")
        print("="*80)
        
        # Print DNN and EKF results
        print(f"\n{'DNN AND EKF MODELS WITH KALMAN FILTER':^100}")
        print("-"*100)
        print(f"{'Method':<20} {'Average Loss':<20} {'Average Loss (degrees)':<25} {'Additional Info':<30}")
        print("-"*100)
        dnn_avg_loss_degrees = dnn_avg_loss * 180 / np.pi
        ekf_avg_loss_degrees = ekf_avg_loss * 180 / np.pi
        additional_info = f"Samples: {dnn_total_samples}, Traj: {len(dnn_trajectory_results)}"
        print(f"{'DNN+Kalman':<20} {dnn_avg_loss:<20.6f} {dnn_avg_loss_degrees:<25.6f} {additional_info:<30}")
        print(f"{'EKF':<20} {ekf_avg_loss:<20.6f} {ekf_avg_loss_degrees:<25.6f} {additional_info:<30}")
        
        # Add comparison row
        dnn_ekf_diff_degrees = dnn_ekf_diff * 180 / np.pi
        if dnn_ekf_diff < 0:
            comparison_text = f"DNN better by {abs(dnn_ekf_diff_degrees):.6f}° ({abs(dnn_ekf_relative_diff):.2f}%)"
        else:
            comparison_text = f"EKF better by {dnn_ekf_diff_degrees:.6f}° ({dnn_ekf_relative_diff:.2f}%)"
        print(f"{'DNN vs EKF':<20} {'Comparison':<20} {comparison_text:<25} {'Performance Gap':<30}")
        
        # Print classic methods results if available
        if classic_methods_avg_losses:
            print(f"\n{'CLASSIC SUBSPACE METHODS':^100}")
            print("-"*100)
            print(f"{'Method':<20} {'Average Loss':<20} {'Average Loss (degrees)':<25} {'Comparison with DNN':<30}")
            print("-"*100)
            
            for method, avg_loss in classic_methods_avg_losses.items():
                diff = dnn_avg_loss - avg_loss
                relative_diff = diff / avg_loss * 100 if avg_loss != 0 else float('inf')
                
                # Convert loss to degrees (assuming loss might be in radians)
                avg_loss_degrees = avg_loss * (180.0 / 3.14159265359)
                
                if diff < 0:
                    comparison = f"DNN better by {abs(diff):.6f} ({abs(relative_diff):.2f}%)"
                elif diff > 0:
                    comparison = f"DNN worse by {diff:.6f} ({relative_diff:.2f}%)"
                else:
                    comparison = "Equal performance"
                    
                print(f"{method:<20} {avg_loss:<20.6f} {avg_loss_degrees:<25.6f} {comparison:<30}")
        
        print("\n" + "="*80)
        
    def _save_results(self) -> None:
        """Save simulation results to the output directory."""
        logger.info(f"Saving results to {self.output_dir}")
        # Placeholder for results saving implementation
        
    def run_scenario(self, scenario_type: str, values: List[Any], full_mode: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Run a parametric scenario with multiple values.
        
        Args:
            scenario_type: Type of scenario (e.g., "SNR", "T", "M", "eta")
            values: List of values to test
            full_mode: If True, run the complete simulation pipeline instead of just training
            
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
            if full_mode:
                result = simulation.run()  # Run complete pipeline
            elif self.config.simulation.evaluate_model and not self.config.simulation.train_model:
                # If in evaluation-only mode, run evaluation instead of training
                logger.info(f"Running evaluation for {scenario_type}={value}")
                result = simulation.run_evaluation()
            else:
                result = simulation.run_training()  # Run training only
                
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
                # Add numpy.scalar to safe globals to fix PyTorch 2.6+ serialization issue
                try:
                    import torch.serialization
                    torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
                    logger.info("Added numpy.scalar to PyTorch safe globals list")
                except Exception as e:
                    logger.warning(f"Could not add numpy.scalar to safe globals: {e}")
                
                # Simple approach - let PyTorch handle it based on version
                state_dict = torch.load(model_path, map_location=device)
            except Exception as e:
                # If any error occurs, try the backward compatibility mode
                logger.warning(f"Standard loading failed, attempting with backward compatibility mode: {e}")
                try:
                    # weights_only=False can help with older PyTorch versions or complex saved states
                    state_dict = torch.load(model_path, map_location=device, weights_only=False)
                except Exception as e2:
                    # Try one more approach with safe_globals context manager
                    try:
                        logger.warning(f"Compatibility mode failed, trying with safe_globals context manager")
                        from torch.serialization import safe_globals
                        with safe_globals(['numpy._core.multiarray.scalar']):
                            state_dict = torch.load(model_path, map_location=device)
                    except Exception as e3:
                        # If all methods fail, raise a comprehensive error
                        raise RuntimeError(f"Failed to load checkpoint file with all methods: {e}, then: {e2}, then: {e3}")

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

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _save_model_state(self, model, model_type=None):
    #     """
    #     Save model state to a timestamped file in the checkpoints directory.
    #     
    #     Args:
    #         model: Model to save
    #         model_type: Type of model for filename (uses config if None)
    #         
    #     Returns:
    #         Path to saved model or None if save failed
    #     """
    #     if model is None:
    #         logger.warning("Cannot save model: model is None")
    #         return None
    #     
    #     model_save_dir = self.output_dir / "checkpoints"
    #     model_save_dir.mkdir(parents=True, exist_ok=True)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     
    #     # Determine model type from config if not provided
    #     if model_type is None:
    #         model_type = self.config.model.type
    #     
    #     model_save_path = model_save_dir / f"saved_{model_type}_{timestamp}.pt"
    #     
    #     logger.info(f"Saving model to {model_save_path}")
    #     
    #     # Save only the state_dict, not the entire model
    #     try:
    #         torch.save(model.state_dict(), model_save_path)
    #         logger.info(f"Model saved successfully to {model_save_path}")
    #         return model_save_path
    #     except Exception as e:
    #         logger.error(f"Failed to save model: {e}")
    #         return None

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _run_single_trajectory_online_learning(self, trajectory_idx: int = 0) -> Dict[str, Any]:
    #     """
    #     Run online learning pipeline for a single trajectory.
    #     """
    #     try:
    #         # Access online learning configuration
    #         online_config = self.config.online_learning
            
            # Check if model is available for online learning
            if self.trained_model is None:
                logger.error("No model available for online learning")
                return {"status": "error", "message": "No model available for online learning"}
            
            # Get online learning parameters
            window_size = getattr(online_config, 'window_size', 10)
            stride = getattr(online_config, 'stride', 5)
            
            # Create an on-demand dataset and dataloader for online learning
            # This will generate data in windows with the current system_model.params.
            # This allows dynamic eta updates within the generator to be reflected.
            system_model_params = self.system_model.params
            
            from simulation.runners.data import create_online_learning_dataset
            from simulation.runners.training import OnlineTrainer
            
            # Create on-demand dataset using the factory function
            logger.info("Online learning is enabled, creating on-demand dataset and dataloader.")
            online_learning_dataset = create_online_learning_dataset(
                system_model_params=system_model_params,
                config=self.config,
                window_size=window_size,
                stride=stride
            )
            
            # Ensure the dataset generator uses the current (reset) eta value
            current_eta = system_model_params.eta
            online_learning_dataset.update_eta(current_eta)
            logger.info(f"Initialized trajectory with eta = {current_eta:.4f}")
            
            # Create dataloader from on-demand dataset
            from torch.utils.data import DataLoader
            self.online_learning_dataloader = DataLoader(
                online_learning_dataset,
                batch_size=1,  # Process one window at a time
                shuffle=False,
                num_workers=0,  # On-demand dataset must use 0 workers
                drop_last=False
            )
            logger.info(f"Created on-demand online learning dataloader for {len(self.online_learning_dataloader)} windows.")
            
            # Create online trainer
            online_trainer = OnlineTrainer(
                model=self.trained_model,
                config=self.config,
                device=device
            )
            
            # Set loss threshold for drift detection
            loss_threshold = getattr(online_config, 'loss_threshold', 0.5)
            max_iterations = getattr(online_config, 'max_iterations', 10)
            
            # Prepare for tracking results
            window_losses = []
            window_covariances = []
            window_update_flags = []
            window_eta_values = []
            window_ekf_predictions = []
            window_ekf_covariances = []
            window_ekf_innovations = []  # New list to track innovations
            window_ekf_kalman_gains = []  # New list to track Kalman gains
            window_ekf_kalman_gain_times_innovation = []  # New list to track K*y
            window_ekf_y_s_inv_y = []  # New list to track y*(S^-1)*y
            window_pre_ekf_losses = []
            window_labels = []  # Store labels for each window
            drift_detected_count = 0
            model_updated_count = 0
            last_ekf_predictions = None  # Track last window's EKF predictions
            last_ekf_covariances = None  # Track last window's EKF covariances
            
            # Process each window of online data
            for window_idx, (time_series_batch, sources_num_batch, labels_batch) in enumerate(tqdm(self.online_learning_dataloader, desc="Online Learning")):
                # --- Dynamic Eta Update Logic ---
                if online_config.eta_update_interval_windows and online_config.eta_update_interval_windows > 0 and \
                   window_idx > 0 and window_idx % online_config.eta_update_interval_windows == 0:
                    current_eta = self.system_model.params.eta # Get current eta from the shared SystemModelParams
                    eta_increment = online_config.eta_increment if online_config.eta_increment is not None else 0.01
                    new_eta = current_eta + eta_increment
                    
                    # Apply min/max constraints if specified
                    if online_config.max_eta is not None:
                        new_eta = min(new_eta, online_config.max_eta)
                    if online_config.min_eta is not None:
                        new_eta = max(new_eta, online_config.min_eta)
                    
                    # Only update if there's an actual change
                    if abs(new_eta - current_eta) > 1e-6:
                        logger.info(f"Online Learning: Dynamically updating eta at window {window_idx}. From {current_eta:.4f} to {new_eta:.4f}")
                        # The dataset holds the generator, which updates the shared self.system_model.params.eta
                        self.online_learning_dataloader.dataset.update_eta(new_eta)
                # --- End Dynamic Eta Update Logic ---
                
                # Unpack batch data
                # The batch data shapes are:
                # time_series_batch: [1, window_size, N, T] = [1, 10, 8, 200]
                # sources_num_batch: [10, 1] (not [1, 10] as expected)
                # labels_batch: [10, 1, 3] (not [1, 10, 3] as expected)
                
                # Extract the content properly from the batch
                time_series_single_window = time_series_batch[0]  # Shape: [window_size, N, T]
                
                # Reshape sources_num to be a list of integers
                sources_num_single_window_list = sources_num_batch[:, 0].tolist() if isinstance(sources_num_batch, torch.Tensor) else [s[0] for s in sources_num_batch]
                
                # Reshape labels to be a list of arrays
                if isinstance(labels_batch, torch.Tensor):
                    # If labels_batch is a tensor [10, 1, 3]
                    labels_single_window_list_of_arrays = [arr[0].cpu().numpy() if isinstance(arr, torch.Tensor) else arr[0] for arr in labels_batch]
                else:
                    # If labels_batch is a list of tensors or arrays
                    labels_single_window_list_of_arrays = [arr[0].cpu().numpy() if isinstance(arr, torch.Tensor) else arr[0] for arr in labels_batch]
                
                # Calculate loss on the current window (generated with current eta)
                current_window_loss, avg_window_cov, ekf_predictions, ekf_covariances, pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y = self._evaluate_window(
                    time_series_single_window, 
                    sources_num_single_window_list, 
                    labels_single_window_list_of_arrays,
                    trajectory_idx,
                    window_idx,
                    is_first_window=(window_idx == 0),
                    last_ekf_predictions=last_ekf_predictions,
                    last_ekf_covariances=last_ekf_covariances
                )
               
                # Update tracking variables
                window_losses.append(current_window_loss)
                window_covariances.append(avg_window_cov)
                window_eta_values.append(self.system_model.params.eta)
                window_ekf_predictions.append(ekf_predictions)
                window_ekf_covariances.append(ekf_covariances)
                window_ekf_innovations.append(ekf_innovations)  # Store innovations
                window_ekf_kalman_gains.append(ekf_kalman_gains)
                window_ekf_kalman_gain_times_innovation.append(ekf_kalman_gain_times_innovation)
                window_ekf_y_s_inv_y.append(ekf_y_s_inv_y)  # Store y*(S^-1)*y
                window_pre_ekf_losses.append(pre_ekf_loss)
                window_labels.append(labels_single_window_list_of_arrays)
                
                # Update last predictions and covariances for next window
                last_ekf_predictions = ekf_predictions
                last_ekf_covariances = ekf_covariances
                
                logger.info(f"Window {window_idx}: Loss = {current_window_loss:.6f}, Cov = {avg_window_cov:.6f} (current eta={self.system_model.params.eta:.4f})")
                
                # Log both pre-EKF and EKF losses for comparison
                logger.info(f"Window {window_idx}: Pre-EKF Loss = {pre_ekf_loss:.6f}, EKF Loss = {current_window_loss:.6f}, Cov = {avg_window_cov:.6f} (eta={self.system_model.params.eta:.4f})")
                
                # Check if loss exceeds threshold (drift detected)
                if current_window_loss > loss_threshold:
                    logger.info(f"Drift detected in window {window_idx} (loss: {current_window_loss:.6f} > threshold: {loss_threshold:.6f})")
                    window_update_flags.append(False)
                else:
                    logger.info(f"No drift detected in window {window_idx} (loss: {current_window_loss:.6f} <= threshold: {loss_threshold:.6f})")
                    window_update_flags.append(False)
            
            # Save final model if it was updated
            if model_updated_count > 0:
                model_save_path = self._save_model_state(
                    self.trained_model,
                    model_type=f"{self.config.model.type}_online_updated"
                )
                logger.info(f"Saved final online-updated model to {model_save_path}")
            
            # Return results
            return {
                "status": "success",
                "online_learning_results": {
                    "window_losses": window_losses,
                    "window_covariances": window_covariances,
                    "window_eta_values": window_eta_values,
                    "window_updates": window_update_flags,
                    "drift_detected_count": drift_detected_count,
                    "model_updated_count": model_updated_count,
                    "window_count": len(self.online_learning_dataloader),
                    "window_size": online_config.window_size,
                    "stride": online_config.stride,
                    "loss_threshold": loss_threshold,
                    "ekf_predictions": window_ekf_predictions,
                    "ekf_covariances": window_ekf_covariances,
                    "ekf_innovations": window_ekf_innovations,  # Add innovations to results
                    "ekf_kalman_gains": window_ekf_kalman_gains,
                    "ekf_kalman_gain_times_innovation": window_ekf_kalman_gain_times_innovation,
                    "ekf_y_s_inv_y": window_ekf_y_s_inv_y,  # Add y*(S^-1)*y to results
                    "pre_ekf_losses": window_pre_ekf_losses,
                    "window_labels": window_labels
                }
            }
            
    #     except Exception as e:
    #         logger.exception(f"Error during online learning: {e}")
    #         return {"status": "error", "message": str(e), "exception": type(e).__name__}

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _run_online_learning(self) -> None:
    #     """
    #     Run online learning pipeline over multiple trajectories and average results.
    #     
    #     This method runs the online learning process over a dataset of trajectories and
    #     averages the results for more robust analysis.
    #     """
    #     try:
    #         # Get dataset size from config
    #         online_config = self.config.online_learning
    #         dataset_size = getattr(online_config, 'dataset_size', 1)
    #         
    #         logger.info(f"Starting online learning over {dataset_size} trajectories")
    #         
    #         all_results = []
    #         
    #         # Run online learning for each trajectory
    #         for trajectory_idx in range(dataset_size):
    #             logger.info(f"Processing trajectory {trajectory_idx + 1}/{dataset_size}")
    #             
    #             # Reset eta to initial value for each new trajectory
    #             initial_eta = 0
    #             logger.info(f"Resetting eta to initial value {initial_eta:.4f} for trajectory {trajectory_idx + 1}")
    #             self.system_model.params.eta = initial_eta
    #             
    #             # Reset the system model's distance noise and eta scaling
    #             self.system_model.eta = self.system_model._SystemModel__set_eta()
    #             if not getattr(self.system_model.params, 'nominal', True):
    #                 self.system_model.location_noise = self.system_model.get_distance_noise(True)
    #             
    #             # Run single trajectory online learning
    #             trajectory_result = self._run_single_trajectory_online_learning(trajectory_idx)
    #             
    #             # Plot single trajectory results
    #             if trajectory_result.get("status") != "error":
    #                 self._plot_single_trajectory_results(trajectory_result["online_learning_results"], trajectory_idx)
    #             
    #             if trajectory_result.get("status") == "error":
    #                 logger.error(f"Error in trajectory {trajectory_idx + 1}: {trajectory_result.get('message')}")
    #                 continue
    #                 
    #             all_results.append(trajectory_result["online_learning_results"])
    #         
    #         if not all_results:
    #             logger.error("No successful trajectory results")
    #             self.results["online_learning_error"] = "No successful trajectory results"
    #             return
    #             
    #         # Average results across all trajectories
    #         averaged_results = self._average_online_learning_results(all_results)
    #         
    #         # Store averaged results
    #         self.results["online_learning"] = averaged_results
    #         
    #         # Plot averaged results
    #         frobenius_norms = self._calculate_frobenius_norm_per_window(averaged_results["ekf_covariances"])
    #         self._plot_online_learning_results(
    #             window_losses=averaged_results["window_losses"],
    #             window_covariances=averaged_results["window_covariances"],
    #             window_eta_values=averaged_results["window_eta_values"],
    #             window_updates=averaged_results["window_updates"],
    #             window_pre_ekf_losses=averaged_results["pre_ekf_losses"],
    #             window_labels=averaged_results["window_labels"],
    #             ekf_covariances=averaged_results["ekf_covariances"],
    #             frobenius_norms=frobenius_norms,
    #             ekf_kalman_gains=averaged_results["ekf_kalman_gains"],
    #             ekf_kalman_gain_times_innovation=averaged_results["ekf_kalman_gain_times_innovation"],
    #             ekf_y_s_inv_y=averaged_results["ekf_y_s_inv_y"]
    #         )
    #         
    #         logger.info(f"Online learning completed over {dataset_size} trajectories: "
    #                    f"{averaged_results['drift_detected_count']:.1f} avg drifts detected, "
    #                    f"{averaged_results['model_updated_count']:.1f} avg model updates")
    #         
    #     except Exception as e:
    #         logger.exception(f"Error during multi-trajectory online learning: {e}")
    #         self.results["online_learning_error"] = str(e)

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _get_optimal_permutation(self, predictions: np.ndarray, true_angles: np.ndarray) -> np.ndarray:
        """
        Calculate optimal permutation between predictions and true angles using RMSPE.
        
        Args:
            predictions: Array of predicted angles [num_sources]
            true_angles: Array of true angles [num_sources]
            
        Returns:
            optimal_perm: Array containing the optimal permutation indices
        """
        import torch
        from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
        from itertools import permutations
        
        # Convert inputs to tensors and reshape
        pred_tensor = torch.tensor(predictions, device=device).view(1, -1)
        true_tensor = torch.tensor(true_angles, device=device).view(1, -1)
        
        num_sources = pred_tensor.shape[1]
        perm = list(permutations(range(num_sources), num_sources))
        num_of_perm = len(perm)
        
        # Calculate errors for all permutations
        err_angle = (pred_tensor[:, perm] - torch.tile(true_tensor[:, None, :], (1, num_of_perm, 1)).to(torch.float32))
        err_angle += torch.pi / 2
        err_angle %= torch.pi
        err_angle -= torch.pi / 2
        rmspe_angle_all_permutations = np.sqrt(1 / num_sources) * torch.linalg.norm(err_angle, dim=-1)
        _, min_idx = torch.min(rmspe_angle_all_permutations, dim=-1)
        
        # Get optimal permutation
        optimal_perm = torch.tensor(perm, dtype=torch.long, device=device)[min_idx]
        return optimal_perm.cpu().numpy()

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _evaluate_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0, 
    #                      is_first_window: bool = True, last_ekf_predictions: List = None, last_ekf_covariances: List = None):
        """
        Calculate loss on a window of trajectory data using Extended Kalman Filter.
        
        Args:
            window_time_series: Time series data for window [batch, window_size, N, T]
            window_sources_num: Source counts for window [batch, window_size]
            window_labels: Labels for window (list of tensors)
            trajectory_idx: Index of the current trajectory
            window_idx: Index of the current window within the trajectory
            is_first_window: Flag indicating if this is the first window evaluation
            last_ekf_predictions: List of last EKF predictions from previous window
            last_ekf_covariances: List of last EKF covariances from previous window
            
        Returns:
            Tuple of (average EKF-filtered loss across window, average covariance across window, 
                     ekf_predictions, ekf_covariances, average pre-EKF loss across window, ekf_innovations)
        """
        # Imports for EKF
        import torch
        from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
        import numpy as np
        
        # Debug: Log input shapes
        logger.debug(f"_evaluate_window input shapes: "
                     f"window_time_series={window_time_series.shape if hasattr(window_time_series, 'shape') else 'not tensor'}, "
                     f"window_sources_num={len(window_sources_num)}, "
                     f"window_labels={len(window_labels)}")
        
        # Unpack arguments directly, assuming they are for a single window
        time_series_steps = window_time_series # Already [window_size, N, T]
        sources_num_per_step = window_sources_num # List[int]
        labels_per_step_list = window_labels    # List[np.ndarray]
        
        # Check that sizes match to avoid index errors
        current_window_len = time_series_steps.shape[0]
        
        # Sanity check the input lengths
        if len(sources_num_per_step) < current_window_len:
            logger.warning(f"Window source count list length ({len(sources_num_per_step)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(sources_num_per_step)
        
        if len(labels_per_step_list) < current_window_len:
            logger.warning(f"Window labels list length ({len(labels_per_step_list)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(labels_per_step_list)
        
        if current_window_len == 0:
            logger.error("Window has zero valid steps. Cannot evaluate.")
            return float('inf'), float('nan'), [], [], float('inf'), []  # Include pre-EKF loss in return
        
        total_loss = 0.0
        num_valid_steps_for_loss = 0
        
        # Add pre-EKF loss tracking for far-field case
        pre_ekf_total_loss = 0.0
        pre_ekf_num_valid_steps = 0
        
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near"
        
        # Use RMSPE loss for evaluation
        rmspe_criterion = RMSPELoss().to(device)
        
        # Initialize Extended Kalman Filters - one for each potential source
        # Use the system's trajectory configuration to determine the appropriate model
        max_sources = self.config.system_model.M
        ekf_filters = []
        
        # Create EKF instances
        for i in range(max_sources):
            ekf_filter = ExtendedKalmanFilter1D.create_from_config(
                self.config, 
                trajectory_type=self.config.trajectory.trajectory_type
            )
            ekf_filters.append(ekf_filter)
        
        # Get current eta value from system model
        current_eta = self.system_model.params.eta
        
        logger.info(f"Initialized {max_sources} EKF instances for window evaluation (eta={current_eta:.4f})")
        
        # Track the EKF predictions, covariances and innovations for each step and source
        ekf_predictions = []
        ekf_covariances = []
        ekf_innovations = []  # New list to track innovations
        ekf_kalman_gains = []  # New list to track Kalman gains
        ekf_kalman_gain_times_innovation = []  # New list to track K*y
        ekf_y_s_inv_y = []  # New list to track y*(S^-1)*y
        
        # Process each step in window
        for step in range(current_window_len):
            try:
                # Extract data for this step
                step_data_tensor = time_series_steps[step:step+1].to(device)  # Shape: [1, N, T] (add batch dim for model)
                num_sources_this_step = sources_num_per_step[step]
                
                # Skip if no sources
                if num_sources_this_step <= 0:
                    ekf_predictions.append([])
                    ekf_covariances.append([])
                    ekf_innovations.append([])
                    ekf_kalman_gains.append([])
                    ekf_kalman_gain_times_innovation.append([])
                    ekf_y_s_inv_y.append([])
                    continue
                
                # Get ground truth labels for this step
                true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
                
                # Forward pass through model
                self.trained_model.eval() # Ensure model is in eval mode for this evaluation part
                with torch.no_grad():
                    if not is_near_field:
                        # Model expects num_sources as int or 0-dim tensor
                        angles_pred, _, _ = self.trained_model(step_data_tensor, num_sources_this_step)
                        
                        # Convert predictions to numpy for EKF processing
                        angles_pred_np = angles_pred.cpu().numpy().flatten()[:num_sources_this_step]
                        
                        # Calculate pre-EKF loss (raw model predictions)
                        pre_ekf_angles_pred = angles_pred.view(1, -1)[:, :num_sources_this_step]
                        true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)
                        
                        # Get optimal permutation for model predictions
                        model_perm = self._get_optimal_permutation(angles_pred_np, true_angles_this_step)
                        angles_pred_np = angles_pred_np[model_perm]
                        pre_ekf_angles_pred = pre_ekf_angles_pred[:, model_perm]
                        
                        # Calculate pre-EKF loss with reordered predictions
                        pre_ekf_loss = rmspe_criterion(pre_ekf_angles_pred, true_angles_tensor)
                        pre_ekf_total_loss += pre_ekf_loss.item()
                        pre_ekf_num_valid_steps += 1
                        
                        # Initialize EKF state if this is the first step
                        if step == 0:
                            if is_first_window:
                                # Initialize with true angles for first window
                                for i in range(num_sources_this_step):
                                    ekf_filters[i].initialize_state(true_angles_this_step[i])
                            else:
                                # Initialize with last predictions from previous window
                                if last_ekf_predictions and len(last_ekf_predictions[-1]) >= num_sources_this_step and \
                                   len(last_ekf_covariances) > 0 and len(last_ekf_covariances[-1]) >= num_sources_this_step:
                                    # Get the last predictions and calculate their optimal permutation
                                    last_predictions_pre_perm = np.array(last_ekf_predictions[-1])[:num_sources_this_step]
                                    last_perm = self._get_optimal_permutation(last_predictions_pre_perm, true_angles_this_step)
                                    last_predictions = last_predictions_pre_perm[last_perm]
                                    
                                    # Get the last covariances and apply the same permutation
                                    last_covariances_pre_perm = np.array(last_ekf_covariances[-1])[:num_sources_this_step]
                                    last_covariances = last_covariances_pre_perm[last_perm]
                                    
                                    for i in range(num_sources_this_step):
                                        ekf_filters[i].initialize_state(last_predictions.flatten()[:num_sources_this_step][i])
                                        ekf_filters[i].P = last_covariances.flatten()[:num_sources_this_step][i]  # Update the covariance matrix
                                else:
                                    # Fallback to true angles if no valid last predictions
                                    logger.warning("No valid last predictions or covariances available, falling back to true angles")
                            for i in range(num_sources_this_step):
                                ekf_filters[i].initialize_state(true_angles_this_step[i])
                        
                        # EKF update for each source
                        step_predictions = []
                        step_covariances = []
                        step_innovations = []  # New list for this step's innovations
                        step_kalman_gains = []  # New list for this step's Kalman gains
                        step_kalman_gain_times_innovation = []  # New list for this step's K*y
                        step_y_s_inv_y = []  # New list for this step's y*(S^-1)*y
                        for i in range(num_sources_this_step):
                            # Predict and update in one step
                            predicted_angle, updated_angle, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y = ekf_filters[i].predict_and_update(
                                measurement=angles_pred_np.flatten()[:num_sources_this_step][i]
                            )
                            
                            # Store prediction, covariance and innovation
                            step_predictions.append(updated_angle)
                            step_covariances.append(ekf_filters[i].P)
                            step_innovations.append(innovation)
                            step_kalman_gains.append(kalman_gain)
                            step_kalman_gain_times_innovation.append(kalman_gain_times_innovation)
                            step_y_s_inv_y.append(y_s_inv_y)
                        
                        # Store predictions, covariances and innovations for this step
                        ekf_predictions.append(step_predictions)
                        ekf_covariances.append(step_covariances)
                        ekf_innovations.append(step_innovations)
                        ekf_kalman_gains.append(step_kalman_gains)
                        ekf_kalman_gain_times_innovation.append(step_kalman_gain_times_innovation)
                        ekf_y_s_inv_y.append(step_y_s_inv_y)
                        
                        # Create tensor from EKF predictions for loss calculation
                        ekf_angles_pred = torch.tensor(step_predictions, device=device).unsqueeze(0)
                        true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)
                        
                        # Calculate loss using EKF predictions
                        loss = rmspe_criterion(ekf_angles_pred, true_angles_tensor)
                    else:
                        # Near-field case - not supported
                        error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                        logger.error(error_msg)
                        raise NotImplementedError(error_msg)
                
                total_loss += loss.item()
                num_valid_steps_for_loss += 1
            except Exception as e:
                logger.warning(f"Error processing step {step}: {e}")
                # Skip this step on error
                continue
        
        # Calculate average loss and covariance for the window
        if num_valid_steps_for_loss > 0:
            avg_loss = total_loss / num_valid_steps_for_loss
            
            # Calculate pre-EKF average loss for far-field case
            avg_pre_ekf_loss = 0.0
            if not is_near_field and pre_ekf_num_valid_steps > 0:
                avg_pre_ekf_loss = pre_ekf_total_loss / pre_ekf_num_valid_steps
            elif is_near_field:
                # Near-field case - not supported
                error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                logger.error(error_msg)
                raise NotImplementedError(error_msg)
            
            # Calculate average covariance across all sources and steps
            avg_window_cov = 0.0
            total_cov_points = 0
            
            if len(ekf_covariances) > 0 and any(len(step_covs) > 0 for step_covs in ekf_covariances):
                # Calculate overall average covariance across all sources and steps
                for step_covs in ekf_covariances:
                    if len(step_covs) > 0:
                        avg_window_cov += sum(step_covs)
                        total_cov_points += len(step_covs)
            
            # Calculate the overall average covariance across all sources and steps
            if total_cov_points > 0:
                avg_window_cov /= total_cov_points
            else:
                avg_window_cov = float('nan')  # No valid covariance points
            
            # Log window summary with columnar format
            self._log_window_summary(avg_pre_ekf_loss, avg_loss, avg_window_cov, current_eta, is_near_field, trajectory_idx, window_idx)
            
            return avg_loss, avg_window_cov, ekf_predictions, ekf_covariances, avg_pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y
        else:
            logger.warning("No valid steps with sources found in the window for loss calculation.")
            return float('inf'), float('nan'), [], [], float('inf'), [], [], [], []  # Include pre-EKF loss in return

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _log_window_summary(
    #     self,
    #     avg_pre_ekf_loss: float,
    #     avg_loss: float,
    #     avg_window_cov: float,
    #     current_eta: float,
    #     is_near_field: bool,
    #     trajectory_idx: int = 0,
    #     window_idx: int = 0
    # ) -> None:
        """
        Log window summary results in a columnar format similar to evaluation results.
        
        Args:
            avg_pre_ekf_loss: Average pre-EKF loss for the window
            avg_loss: Average EKF loss for the window
            avg_window_cov: Average covariance for the window
            current_eta: Current eta value
            is_near_field: Whether this is near field scenario
            trajectory_idx: Index of the current trajectory
            window_idx: Index of the current window within the trajectory
        """
        print(f"\n{'WINDOW SUMMARY - WINDOW ' + str(window_idx) + ' TRAJECTORY ' + str(trajectory_idx):^100}")
        print("-"*100)
        print(f"{'Metric':<20} {'Loss Value':<20} {'Loss (degrees)':<25} {'Additional Info':<30}")
        print("-"*100)
        
        if not is_near_field:
            # Convert losses to degrees
            pre_ekf_loss_degrees = avg_pre_ekf_loss * 180 / np.pi
            ekf_loss_degrees = avg_loss * 180 / np.pi
            loss_difference_degrees = ekf_loss_degrees - pre_ekf_loss_degrees  # Negative = EKF better
            
            # Determine performance status
            ekf_improved = loss_difference_degrees < 0
            abs_difference = abs(loss_difference_degrees)
            improvement_percent = abs_difference / pre_ekf_loss_degrees * 100
            best_method = "EKF" if ekf_improved else "SubspaceNet"
            best_loss_degrees = min(pre_ekf_loss_degrees, ekf_loss_degrees)
            
            # Display individual losses
            print(f"{'SubspaceNet Loss':<20} {avg_pre_ekf_loss:<20.6f} {pre_ekf_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
            print(f"{'EKF Loss':<20} {avg_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'Avg Cov: {avg_window_cov:.2e}, w: {window_idx}, t: {trajectory_idx}':<30}")
            
            # Display comparison result
            if ekf_improved:
                status_icon = "✓"
                status_text = "EKF OVERCOMES SubspaceNet"
                change_text = f"↓ {abs_difference:.4f}° ({improvement_percent:.1f}% better)"
            else:
                status_icon = "✗"
                status_text = "SubspaceNet BETTER than EKF"
                change_text = f"↑ {abs_difference:.4f}° ({improvement_percent:.1f}% worse)"
            
            print(f"{'WINNER':<20} {best_method:<20} {best_loss_degrees:<25.6f} {status_icon + ' ' + status_text:<30}")
            print(f"{'Performance':<20} {change_text:<45} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
            print("-" * 100)
        else:
            # Near field - only EKF loss (no SubspaceNet comparison available)
            ekf_loss_degrees = avg_loss * 180 / np.pi
            print(f"{'EKF Loss':<20} {avg_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
            print(f"{'Mode':<20} {'NEAR FIELD':<20} {'(No SubspaceNet comparison)':<25} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
            print("-" * 100)

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _plot_online_learning_results(self, window_losses, window_covariances, window_eta_values, window_updates, window_pre_ekf_losses, window_labels, ekf_covariances, frobenius_norms, ekf_kalman_gains=None, ekf_kalman_gain_times_innovation=None, ekf_y_s_inv_y=None):
        """
        Plot online learning results including plots as a function of eta.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import datetime
            
            # Create timestamp and plot directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = self.output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            def set_adjusted_ylim(ax, data, padding=0.1):
                """Helper function to set y limits excluding first point"""
                if len(data) > 1:
                    data_without_first = data[1:]
                    ymin = min(data_without_first)
                    ymax = max(data_without_first)
                    range_y = ymax - ymin
                    ax.set_ylim([ymin - range_y * padding, ymax + range_y * padding])
            
            # Find indices where eta changes
            eta_changes = []
            eta_values = []
            for i in range(1, len(window_eta_values)):
                if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
                    eta_changes.append(i)
                    eta_values.append(window_eta_values[i])
            
            # Create figure with multiple subplots (4x2 layout for 8 plots)
            fig = plt.figure(figsize=(20, 20))
            
            # 1. Plot loss vs window index
            ax1 = fig.add_subplot(4, 2, 1)
            x = np.arange(len(window_losses))[1:]  # Start from second sample
            ax1.plot(x, np.array(window_losses)[1:], 'b-', marker='o', label='EKF Loss')
            ax1.plot(x, np.array(window_pre_ekf_losses)[1:], 'r-', marker='s', label='SubspaceNet Loss')
            
            # Set y-axis limit to 0.14 for loss plot only
            ax1.set_ylim([None, 0.14])
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax1.text(idx, 0.13, f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
            ax1.set_xlabel('Window Index')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss vs Window Index (Starting from Window 1)\nRMSPE = √(1/N * Σ(θ_pred - θ_true)²)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Plot EKF improvement vs window index (reversed: SubspaceNet - EKF)
            ax2 = fig.add_subplot(4, 2, 2)
            x = np.arange(len(window_losses))[1:]  # Start from second sample
            improvement = np.array(window_pre_ekf_losses)[1:] - np.array(window_losses)[1:]  # Reversed calculation, starting from second sample
            ax2.plot(x, improvement, 'g-', marker='o')
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax2.text(idx, ax2.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
            ax2.set_xlabel('Window Index')
            ax2.set_ylabel('Loss Difference')
            ax2.set_title('SubspaceNet Loss - EKF Loss vs Window Index (Starting from Window 1)\nImprovement = L_SubspaceNet - L_EKF')
            ax2.grid(True, alpha=0.3)
            
            # 3. Plot covariance vs window index
            ax3 = fig.add_subplot(4, 2, 3)
            x = np.arange(len(window_covariances))[1:]  # Start from second sample
            ax3.plot(x, np.array(window_covariances)[1:], 'g-', marker='o', label='Average Covariance')
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax3.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax3.text(idx, max(np.array(window_covariances)[1:]), f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Highlight windows where model was updated (adjusted for starting from second sample)
            if window_updates:
                update_indices = [i for i, updated in enumerate(window_updates) if updated and i >= 1]
                update_covs = [window_covariances[i] for i in update_indices]
                ax3.scatter(update_indices, update_covs, color='r', s=80, marker='o', label='Model Updated')
            
            # Add labels and title
            ax3.set_xlabel('Window Index')
            ax3.set_ylabel('Average Covariance')
            ax3.set_title('Covariance vs Window Index (Starting from Window 1)\nP_k|k = (I - K_k H) P_k|k-1')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Plot average innovation magnitude vs window index
            ax4 = fig.add_subplot(4, 2, 4)
            x = np.arange(len(window_losses))[1:]  # Start from second sample
            
            # Calculate average innovation magnitude per window from the results
            avg_innovations = []
            if "ekf_innovations" in self.results.get("online_learning", {}):
                ekf_innovations = self.results["online_learning"]["ekf_innovations"]
                for window_innovations in ekf_innovations:
                    window_avg = []
                    for step_innovations in window_innovations:
                        if step_innovations:  # Check if there are any innovations in this step
                            window_avg.extend([abs(inn) for inn in step_innovations])  # Use absolute values
                    if window_avg:
                        avg_innovations.append(np.mean(window_avg))
                    else:
                        avg_innovations.append(0)
            else:
                # Fallback: create zeros if no innovation data available
                avg_innovations = [0] * len(window_losses)
            
            # Ensure we have the right number of points and start from second sample
            if len(avg_innovations) != len(window_losses):
                avg_innovations = avg_innovations[:len(window_losses)] if len(avg_innovations) > len(window_losses) else avg_innovations + [0] * (len(window_losses) - len(avg_innovations))
            
            ax4.plot(x, np.array(avg_innovations)[1:], 'b-', marker='o', linewidth=2, markersize=6, label='Average Innovation Magnitude')
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax4.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax4.text(idx, ax4.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
            ax4.set_xlabel('Window Index')
            ax4.set_ylabel('Average Innovation Magnitude')
            ax4.set_title('EKF Innovation Magnitude vs Window Index (Starting from Window 1)\nInnovation = |z_k - H x̂_k|k-1| = |measurement - prediction|')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Plot average Kalman gain vs window index
            if ekf_kalman_gains is not None:
                ax5 = fig.add_subplot(4, 2, 5)
                # Calculate average Kalman gain per window
                avg_kalman_gains = []
                for window_gains in ekf_kalman_gains:
                    window_avg = []
                    for step_gains in window_gains:
                        if step_gains:  # Check if there are any gains in this step
                            window_avg.extend(step_gains)
                    if window_avg:
                        avg_kalman_gains.append(np.mean(window_avg))
                    else:
                        avg_kalman_gains.append(0)
                
                x = np.arange(len(avg_kalman_gains))[1:]  # Start from second sample
                ax5.plot(x, np.array(avg_kalman_gains)[1:], 'purple', marker='d', label='Average Kalman Gain')
                
                # Add eta change markers (adjusted for starting from second sample)
                for idx, eta in zip(eta_changes, eta_values):
                    if idx >= 1:  # Only show markers from second sample onwards
                        ax5.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                        ax5.text(idx, ax5.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
                ax5.set_xlabel('Window Index')
                ax5.set_ylabel('Average Kalman Gain')
                ax5.set_title('Average Kalman Gain vs Window Index (Starting from Window 1)\nK_k = P_k|k-1 H^T (H P_k|k-1 H^T + R)^-1')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Plot average K*y vs window index
            if ekf_kalman_gain_times_innovation is not None:
                ax6 = fig.add_subplot(4, 2, 6)
                # Calculate average K*y per window
                avg_k_times_y = []
                for window_k_times_y in ekf_kalman_gain_times_innovation:
                    window_avg = []
                    for step_k_times_y in window_k_times_y:
                        if step_k_times_y:  # Check if there are any values in this step
                            window_avg.extend(step_k_times_y)
                    if window_avg:
                        avg_k_times_y.append(np.mean(window_avg))
                    else:
                        avg_k_times_y.append(0)
                
                x = np.arange(len(avg_k_times_y))[1:]  # Start from second sample
                ax6.plot(x, np.array(avg_k_times_y)[1:], 'orange', marker='v', label='Average K*Innovation')
                
                # Add eta change markers (adjusted for starting from second sample)
                for idx, eta in zip(eta_changes, eta_values):
                    if idx >= 1:  # Only show markers from second sample onwards
                        ax6.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                        ax6.text(idx, ax6.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
                ax6.set_xlabel('Window Index')
                ax6.set_ylabel('Average K*Innovation')
                ax6.set_title('Average Kalman Gain × Innovation vs Window Index (Starting from Window 1)\nK_k × ν_k = K_k × (z_k - H x̂_k|k-1)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # 7. Plot average y*(S^-1)*y vs window index
            if ekf_y_s_inv_y is not None:
                ax7 = fig.add_subplot(4, 2, 7)
                # Calculate average y*(S^-1)*y per window
                avg_y_s_inv_y = []
                for window_y_s_inv_y in ekf_y_s_inv_y:
                    window_avg = []
                    for step_y_s_inv_y in window_y_s_inv_y:
                        if step_y_s_inv_y:  # Check if there are any values in this step
                            window_avg.extend(step_y_s_inv_y)
                    if window_avg:
                        avg_y_s_inv_y.append(np.mean(window_avg))
                    else:
                        avg_y_s_inv_y.append(0)
                
                x = np.arange(len(avg_y_s_inv_y))[1:]  # Start from second sample
                ax7.plot(x, np.array(avg_y_s_inv_y)[1:], 'red', marker='^', label='Average y*(S^-1)*y')
                
                # Add eta change markers (adjusted for starting from second sample)
                for idx, eta in zip(eta_changes, eta_values):
                    if idx >= 1:  # Only show markers from second sample onwards
                        ax7.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                        ax7.text(idx, ax7.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
                
                # Add labels and title
                ax7.set_xlabel('Window Index')
                ax7.set_ylabel('Average y*(S^-1)*y')
                ax7.set_title('Average Innovation Covariance Metric vs Window Index (Starting from Window 1)\ny*(S^-1)*y = ν^T S^-1 ν')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = plot_dir / f"online_learning_results_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Plot Frobenius norms with adjusted y-axis
            self._plot_frobenius_norms(frobenius_norms, window_eta_values, plot_dir, timestamp, exclude_first=True)
            
            # Plot online learning trajectory
            self._plot_online_learning_trajectory(window_labels, plot_dir, timestamp)
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting online learning results: {e}")
            logger.debug("Error details:", exc_info=True)
    
    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _plot_single_trajectory_results(self, trajectory_results, trajectory_idx):
        """
        Plot results for a single trajectory including loss difference and innovation.
        
        Args:
            trajectory_results: Dictionary containing single trajectory results
            trajectory_idx: Index of the trajectory being plotted
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import datetime
            
            # Create timestamp and plot directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = self.output_dir / "plots" / "single_trajectories"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract data from trajectory results
            pre_ekf_losses = np.array(trajectory_results['pre_ekf_losses'])  # Shape: (30,)
            window_losses = np.array(trajectory_results['window_losses'])    # Shape: (30,) - post EKF losses
            ekf_innovations = np.array(trajectory_results['ekf_innovations']) # Shape: (30, 5, 3)
            window_eta_values = trajectory_results['window_eta_values']
            
            # Calculate loss difference: pre_ekf_loss - post_ekf_loss (positive means EKF improved)
            loss_difference = pre_ekf_losses - window_losses
            
            # Calculate average innovation magnitude per window
            # Aggregate across steps (5) and sources (3) to get one value per window (30)
            avg_innovations_per_window = []
            for window_idx in range(ekf_innovations.shape[0]):  # 30 windows
                window_innovations = []
                for step_idx in range(ekf_innovations.shape[1]):  # 5 steps
                    for source_idx in range(ekf_innovations.shape[2]):  # 3 sources
                        innovation_val = ekf_innovations[window_idx, step_idx, source_idx]
                        window_innovations.append(abs(innovation_val))  # Use absolute value
                
                if window_innovations:
                    avg_innovations_per_window.append(np.mean(window_innovations))
                else:
                    avg_innovations_per_window.append(0)
            
            avg_innovations_per_window = np.array(avg_innovations_per_window)
            
            # Find indices where eta changes
            eta_changes = []
            eta_values = []
            for i in range(1, len(window_eta_values)):
                if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
                    eta_changes.append(i)
                    eta_values.append(window_eta_values[i])
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Window indices
            window_indices = np.arange(len(window_losses))
            
            # Plot 1: Loss difference (Pre-EKF Loss - Post-EKF Loss)
            ax1.plot(window_indices, loss_difference, 'g-', marker='o', linewidth=2, markersize=6, label='Pre-EKF Loss - Post-EKF Loss')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Improvement Line')
            
            # Add eta change markers
            for idx, eta in zip(eta_changes, eta_values):
                ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax1.text(idx, ax1.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax1.set_xlabel('Window Index')
            ax1.set_ylabel('Loss Difference')
            ax1.set_title(f'Trajectory {trajectory_idx}: Pre-EKF Loss - Post-EKF Loss vs Window Index\n(Positive = EKF Improved, Negative = EKF Worse)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Innovation magnitude
            ax2.plot(window_indices, avg_innovations_per_window, 'b-', marker='s', linewidth=2, markersize=6, label='Average Innovation Magnitude')
            
            # Add eta change markers
            for idx, eta in zip(eta_changes, eta_values):
                ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax2.text(idx, ax2.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax2.set_xlabel('Window Index')
            ax2.set_ylabel('Average Innovation Magnitude')
            ax2.set_title(f'Trajectory {trajectory_idx}: EKF Innovation Magnitude vs Window Index\n|Innovation| = |z_k - H x̂_k|k-1| = |measurement - prediction|')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = plot_dir / f"single_trajectory_{trajectory_idx}_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Single trajectory plot saved: {plot_path}")
            
        except ImportError:
            logger.warning("matplotlib not available for single trajectory plotting")
        except Exception as e:
            logger.error(f"Error plotting single trajectory results for trajectory {trajectory_idx}: {e}")
            logger.debug("Error details:", exc_info=True)
    
    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _plot_online_learning_trajectory(self, window_labels, plot_dir, timestamp):
        """
        Plot the full trajectory across all windows for online learning.
        
        Args:
            window_labels: List of labels for each window, where each window contains a list of numpy arrays
            plot_dir: Directory to save the plots
            timestamp: Timestamp for the plot filename
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Flatten all labels across all windows to get the complete trajectory
            all_angles = []
            all_distances = []
            window_indices = []  # Track which window each step belongs to
            
            for window_idx, window_label_list in enumerate(window_labels):
                for step_idx, step_labels in enumerate(window_label_list):
                    # step_labels is a numpy array containing [angle1, angle2, ..., angleM] (in radians)
                    # For far-field, we assume distances are constant (e.g., 30m)
                    # For near-field, distances would be included in the labels
                    
                    # Convert radians to degrees for plotting
                    angles_deg = step_labels * (180.0 / np.pi)
                    all_angles.append(angles_deg)
                    
                    # Use the same range increment mechanism as in generate_trajectories
                    # Start at 20m and increment by 1m each step
                    global_step = len(all_angles) - 1  # Current global step index
                    base_distance = 20.0 + global_step * 1.0  # Increment by 1m each step
                    
                    num_sources = len(step_labels)
                    distances = np.full(num_sources, base_distance)
                    all_distances.append(distances)
                    
                    # Track window index
                    window_indices.append(window_idx)
            
            if not all_angles:
                logger.warning("No trajectory data available for plotting")
                return
            
            # Find the maximum number of sources across all steps
            max_sources = max(len(angles) for angles in all_angles)
            total_steps = len(all_angles)
            
            # Pad arrays to have consistent dimensions
            padded_angles = np.full((total_steps, max_sources), np.nan)
            padded_distances = np.full((total_steps, max_sources), np.nan)
            
            for step_idx, (angles, distances) in enumerate(zip(all_angles, all_distances)):
                num_sources = len(angles)
                padded_angles[step_idx, :num_sources] = angles
                padded_distances[step_idx, :num_sources] = distances
            
            # Create the trajectory plot
            plt.figure(figsize=(12, 10))
            
            # Plot each source trajectory
            for s in range(max_sources):
                # Get valid data for this source (some steps might have fewer sources)
                valid_mask = ~np.isnan(padded_angles[:, s])
                if np.any(valid_mask):
                    angles_rad = padded_angles[valid_mask, s] * (np.pi / 180.0)  # Convert back to radians for plotting
                    distances = padded_distances[valid_mask, s]
                    
                    # Convert from polar to Cartesian coordinates
                    x = distances * np.cos(angles_rad)
                    y = distances * np.sin(angles_rad)
                    
                    # Plot trajectory
                    plt.plot(x, y, '-o', markersize=4, label=f'Source {s+1}')
                    
                    # Mark start and end points
                    if len(x) > 0:
                        plt.plot(x[0], y[0], 'go', markersize=8)  # Green for start
                        plt.plot(x[-1], y[-1], 'ro', markersize=8)  # Red for end
            
            # Plot radar location
            plt.plot(0, 0, 'bD', markersize=12, label='Radar')
            
            # Add distance circles
            for d in [20, 30, 40, 50]:
                circle = plt.Circle((0, 0), d, fill=False, linestyle='--', alpha=0.3)
                plt.gca().add_patch(circle)
                plt.text(0, d, f'{d}m', va='bottom', ha='center')
            
            # Add angle lines
            for a in range(-90, 91, 30):
                a_rad = a * (np.pi/180)
                plt.plot([0, 60*np.cos(a_rad)], [0, 60*np.sin(a_rad)], 'k:', alpha=0.2)
                plt.text(55*np.cos(a_rad), 55*np.sin(a_rad), f'{a}°', 
                        va='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))
            
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.title(f'Online Learning Full Trajectory (T={total_steps}, Sources={max_sources}, Windows={len(window_labels)})')
            plt.legend()
            
            # Add window boundary markers if there are multiple windows
            if len(window_labels) > 1:
                # Find window boundaries
                window_boundaries = []
                current_window = window_indices[0]
                for i, window_idx in enumerate(window_indices):
                    if window_idx != current_window:
                        window_boundaries.append(i)
                        current_window = window_idx
                
                # Add vertical lines for window boundaries
                for boundary_idx in window_boundaries:
                    if boundary_idx < len(all_angles):
                        # Get the position at the boundary
                        angles_at_boundary = all_angles[boundary_idx]
                        distances_at_boundary = all_distances[boundary_idx]
                        
                        # Plot boundary markers for each source
                        for s in range(len(angles_at_boundary)):
                            if not np.isnan(angles_at_boundary[s]):
                                angle_rad = angles_at_boundary[s] * (np.pi / 180.0)
                                distance = distances_at_boundary[s]
                                x = distance * np.cos(angle_rad)
                                y = distance * np.sin(angle_rad)
                                plt.plot(x, y, 'ks', markersize=10, markerfacecolor='none', markeredgewidth=2)
            
            # Save the plot
            plot_path = plot_dir / f"online_learning_trajectory_{timestamp}.png"
            plt.savefig(plot_path)
            logger.info(f"Online learning trajectory plot saved to {plot_dir}:")
            logger.info(f"  - Trajectory plot: {plot_path.name}")
            
        except ImportError:
            logger.warning("matplotlib not available for plotting online learning trajectory")
        except Exception as e:
            logger.error(f"Error plotting online learning trajectory: {e}")
    
    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _calculate_frobenius_norm_per_window(self, window_ekf_covariances):
    #     """
    #     Calculate the Frobenius norm of covariance matrices per window.
    #     
    #     Args:
    #         window_ekf_covariances: List of covariances per window
    #                                Shape: [num_windows][num_steps_in_window][num_sources]
    #                                Each covariance is a scalar or matrix
    #     
    #     Returns:
    #         numpy.ndarray: Frobenius norms per window, shape [num_windows, 1]
    #     """
    #     try:
    #         import numpy as np
    #         
    #         frobenius_norms = []
    #         
    #         for window_idx, window_covs in enumerate(window_ekf_covariances):
    #             window_frobenius_sum = 0.0
    #             valid_steps = 0
    #             
    #             for step_idx, step_covs in enumerate(window_covs):
    #                 if len(step_covs) > 0:  # Check if there are covariances for this step
    #                     for source_idx, cov in enumerate(step_covs):
    #                         if isinstance(cov, (float, int)):
    #                             # Scalar covariance - treat as 1x1 matrix
    #                             window_frobenius_sum += abs(cov) ** 2
    #                         else:
    #                             # Matrix covariance - calculate Frobenius norm
    #                             if hasattr(cov, 'shape'):
    #                                 window_frobenius_sum += np.sum(np.abs(cov) ** 2)
    #                             else:
    #                                 window_frobenius_sum += abs(cov) ** 2
    #                     valid_steps += 1
    #             
    #             # Calculate average Frobenius norm for this window
    #             if valid_steps > 0:
    #                 avg_frobenius = np.sqrt(window_frobenius_sum / valid_steps)
    #             else:
    #                 avg_frobenius = 0.0
    #             
    #             frobenius_norms.append([avg_frobenius])
    #         
    #         return np.array(frobenius_norms)  # Shape: [num_windows, 1]
    #         
    #     except Exception as e:
    #         logger.error(f"Error calculating Frobenius norm per window: {e}")
    #         # Return zeros if calculation fails
    #         num_windows = len(window_ekf_covariances)
    #         return np.zeros((num_windows, 1))

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _plot_frobenius_norms(self, frobenius_norms, window_eta_values, plot_dir, timestamp, exclude_first=False):
    #     """
    #     Plot Frobenius norms of covariance matrices per window.
    #     
    #     Args:
    #         frobenius_norms: List of Frobenius norms per window
    #         window_eta_values: List of eta values per window
    #         plot_dir: Directory to save the plots
    #         timestamp: Timestamp for the plot filename
    #         exclude_first: Whether to exclude the first point from the plot
    #     """
    #     try:
    #         import matplotlib.pyplot as plt
    #         import numpy as np
    #         
    #         def set_adjusted_ylim(ax, data, padding=0.1):
    #             """Helper function to set y limits excluding first point"""
    #             if len(data) > 1:
    #                 data_without_first = data[1:]
    #                 ymin = min(data_without_first)
    #                 ymax = max(data_without_first)
    #                 range_y = ymax - ymin
    #                 ax.set_ylim([ymin - range_y * padding, ymax + range_y * padding])
    #         
    #         # Find indices where eta changes
    #         eta_changes = []
    #         eta_values = []
    #         for i in range(1, len(window_eta_values)):
    #             if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
    #                 eta_changes.append(i)
    #                 eta_values.append(window_eta_values[i])
    #         
    #         # Create figure for Frobenius norms
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    #         
    #         # Plot Frobenius norm vs window index
    #         x = np.arange(len(frobenius_norms))
    #         ax1.plot(x, frobenius_norms, 'b-', marker='o', linewidth=2, markersize=6, label='Frobenius Norm')
    #         
    #         # Add eta change markers
    #         for idx, eta in zip(eta_changes, eta_values):
    #             ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
    #             ax1.text(idx, max(frobenius_norms[1:]), f'η={eta:.3f}', rotation=90, verticalalignment='top')
    #         
    #         # Set y-axis limits excluding first point if requested
    #         if exclude_first:
    #             set_adjusted_ylim(ax1, frobenius_norms)
    #         
    #         # Add labels and title
    #         ax1.set_xlabel('Window Index')
    #         ax1.set_ylabel('Frobenius Norm')
    #         ax1.set_title('Covariance Matrix Frobenius Norm vs Window Index' + 
    #                      (' (Excluding Initial Value)' if exclude_first else ''))
    #         ax1.legend()
    #         ax1.grid(True, alpha=0.3)
    #         
    #         # Plot Frobenius norm vs eta
    #         ax2.plot(window_eta_values, frobenius_norms, 'b-', marker='o', linewidth=2, markersize=6, label='Frobenius Norm')
    #         
    #         # Add eta change markers
    #         for eta in eta_values:
    #             ax2.scatter([eta], [frobenius_norms[window_eta_values.index(eta)]], color='red', marker='*', s=200, zorder=5)
    #             ax2.text(eta, max(frobenius_norms[1:]), f'η={eta:.3f}', rotation=90, verticalalignment='top')
    #         
    #         # Set y-axis limits excluding first point if requested
    #         if exclude_first:
    #             set_adjusted_ylim(ax2, frobenius_norms)
    #         
    #         # Add labels and title
    #         ax2.set_xlabel('Eta Value')
    #         ax2.set_ylabel('Frobenius Norm')
    #         ax2.set_title('Covariance Matrix Frobenius Norm vs Eta' + 
    #                      (' (Excluding Initial Value)' if exclude_first else ''))
    #         ax2.legend()
    #         ax2.grid(True, alpha=0.3)
    #         
    #         # Adjust layout and save
    #         plt.tight_layout()
    #         plot_path = plot_dir / f"frobenius_norms_{timestamp}.png"
    #         plt.savefig(plot_path)
    #         plt.close()
    #         
    #         logger.info(f"Frobenius norm plots saved to {plot_path}")
    #         
    #     except ImportError:
    #         logger.warning("matplotlib not available for plotting Frobenius norms")
    #     except Exception as e:
    #         logger.error(f"Error plotting Frobenius norms: {e}")

    # MOVED TO OnlineLearning class in simulation/runners/Online_learning.py
    # def _average_online_learning_results(self, results_list):
    #     """
    #     Average results across multiple trajectories.
    #     
    #     Args:
    #         results_list: List of dictionaries containing results from each trajectory
    #     
    #     Returns:
    #         Dictionary with averaged results
    #     """
    #     import numpy as np
    #         
    #     # Initialize lists to store results from all trajectories
    #     all_window_losses = []
    #     all_window_covariances = []
    #     all_window_eta_values = []
    #     all_window_updates = []
    #     all_drift_detected = []
    #     all_model_updated = []
    #     all_window_count = []
    #     all_window_size = []
    #     all_stride = []
    #     all_loss_threshold = []
    #     all_ekf_predictions = []
    #     all_ekf_covariances = []
    #     all_ekf_innovations = []  # New list for innovations
    #     all_ekf_kalman_gains = []  # New list for Kalman gains
    #     all_ekf_kalman_gain_times_innovation = []  # New list for K*y
    #     all_ekf_y_s_inv_y = []  # New list for y*(S^-1)*y
    #     all_pre_ekf_losses = []
    #     all_window_labels = []
    #     
    #     # Collect results from each trajectory
    #     for result in results_list:
    #         all_window_losses.append(result["window_losses"])
    #         all_window_covariances.append(result["window_covariances"])
    #         all_window_eta_values.append(result["window_eta_values"])
    #         all_window_updates.append(result["window_updates"])
    #         all_drift_detected.append(result["drift_detected_count"])
    #         all_model_updated.append(result["model_updated_count"])
    #         all_window_count.append(result["window_count"])
    #         all_window_size.append(result["window_size"])
    #         all_stride.append(result["stride"])
    #         all_loss_threshold.append(result["loss_threshold"])
    #         all_ekf_predictions.append(result["ekf_predictions"])
    #         all_ekf_covariances.append(result["ekf_covariances"])
    #         all_ekf_innovations.append(result["ekf_innovations"])  # Collect innovations
    #         all_ekf_kalman_gains.append(result["ekf_kalman_gains"])
    #         all_ekf_kalman_gain_times_innovation.append(result["ekf_kalman_gain_times_innovation"])
    #         all_ekf_y_s_inv_y.append(result["ekf_y_s_inv_y"])  # Collect y*(S^-1)*y
    #         all_pre_ekf_losses.append(result["pre_ekf_losses"])
    #         all_window_labels.append(result["window_labels"])
    #     
    #     # Average numerical results
    #     avg_window_losses = np.mean(all_window_losses, axis=0)
    #     avg_window_covariances = np.mean(all_window_covariances, axis=0)
    #     avg_window_eta_values = all_window_eta_values[0]  # Should be same for all trajectories
    #     avg_window_updates = np.mean(all_window_updates, axis=0)
    #     avg_drift_detected = np.mean(all_drift_detected)
    #     avg_model_updated = np.mean(all_model_updated)
    #     avg_pre_ekf_losses = np.mean(all_pre_ekf_losses, axis=0)
    #     
    #     # Average innovations - handle nested structure
    #     avg_ekf_innovations = []
    #     for window_idx in range(len(all_ekf_innovations[0])):  # For each window
    #         window_innovations = []
    #         for step_idx in range(len(all_ekf_innovations[0][window_idx])):  # For each step
    #             step_innovations = []
    #             for source_idx in range(len(all_ekf_innovations[0][window_idx][step_idx])):  # For each source
    #                 innovations = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_innovations]
    #                 step_innovations.append(np.mean(innovations))
    #             window_innovations.append(step_innovations)
    #         avg_ekf_innovations.append(window_innovations)
    #     
    #     # Average Kalman gains - handle nested structure
    #     avg_ekf_kalman_gains = []
    #     for window_idx in range(len(all_ekf_kalman_gains[0])):  # For each window
    #         window_kalman_gains = []
    #         for step_idx in range(len(all_ekf_kalman_gains[0][window_idx])):  # For each step
    #             step_kalman_gains = []
    #             for source_idx in range(len(all_ekf_kalman_gains[0][window_idx][step_idx])):  # For each source
    #                 kalman_gains = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_kalman_gains]
    #                 step_kalman_gains.append(np.mean(kalman_gains))
    #             window_kalman_gains.append(step_kalman_gains)
    #         avg_ekf_kalman_gains.append(window_kalman_gains)
    #     
    #     # Average K*y - handle nested structure
    #     avg_ekf_kalman_gain_times_innovation = []
    #     for window_idx in range(len(all_ekf_kalman_gain_times_innovation[0])):  # For each window
    #         window_k_times_y = []
    #         for step_idx in range(len(all_ekf_kalman_gain_times_innovation[0][window_idx])):  # For each step
    #             step_k_times_y = []
    #             for source_idx in range(len(all_ekf_kalman_gain_times_innovation[0][window_idx][step_idx])):  # For each source
    #                 k_times_y = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_kalman_gain_times_innovation]
    #                 step_k_times_y.append(np.mean(k_times_y))
    #             window_k_times_y.append(step_k_times_y)
    #         avg_ekf_kalman_gain_times_innovation.append(window_k_times_y)
    #     
    #     # Average y*(S^-1)*y - handle nested structure
    #     avg_ekf_y_s_inv_y = []
    #     for window_idx in range(len(all_ekf_y_s_inv_y[0])):  # For each window
    #         window_y_s_inv_y = []
    #         for step_idx in range(len(all_ekf_y_s_inv_y[0][window_idx])):  # For each step
    #             step_y_s_inv_y = []
    #             for source_idx in range(len(all_ekf_y_s_inv_y[0][window_idx][step_idx])):  # For each source
    #                 y_s_inv_y = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_y_s_inv_y]
    #                 step_y_s_inv_y.append(np.mean(y_s_inv_y))
    #             window_y_s_inv_y.append(step_y_s_inv_y)
    #         avg_ekf_y_s_inv_y.append(window_y_s_inv_y)
    #     
    #     return {
    #         "window_losses": avg_window_losses.tolist(),
    #         "window_covariances": avg_window_covariances.tolist(),
    #         "window_eta_values": avg_window_eta_values,
    #         "window_updates": avg_window_updates.tolist(),
    #         "drift_detected_count": float(avg_drift_detected),
    #         "model_updated_count": float(avg_model_updated),
    #         "window_count": all_window_count[0],  # Should be same for all trajectories
    #         "window_size": all_window_size[0],    # Should be same for all trajectories
    #         "stride": all_stride[0],              # Should be same for all trajectories
    #         "loss_threshold": all_loss_threshold[0],  # Should be same for all trajectories
    #         "ekf_predictions": all_ekf_predictions[0],  # Take first trajectory's predictions
    #         "ekf_covariances": all_ekf_covariances[0],  # Take first trajectory's covariances
    #         "ekf_innovations": avg_ekf_innovations,  # Add averaged innovations
    #         "ekf_kalman_gains": avg_ekf_kalman_gains,  # Add averaged Kalman gains
    #         "ekf_kalman_gain_times_innovation": avg_ekf_kalman_gain_times_innovation,  # Add averaged K*y
    #         "ekf_y_s_inv_y": avg_ekf_y_s_inv_y,  # Add averaged y*(S^-1)*y
    #         "pre_ekf_losses": avg_pre_ekf_losses.tolist(),
    #         "window_labels": all_window_labels[0]  # Take first trajectory's labels
    #     }
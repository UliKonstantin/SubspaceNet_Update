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
from utils.utils import save_model_state
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
                save_model_state(self.trained_model, self.output_dir, f"{self.config.model.type}_trained")
            
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
        
        # Check if model paths are provided in config for this scenario
        model_paths = None
        if hasattr(self.config, 'scenario_config') and hasattr(self.config.scenario_config, 'model_paths') and self.config.scenario_config.model_paths:
            model_paths = self.config.scenario_config.model_paths
            logger.info(f"Found {len(model_paths)} model paths in scenario_config for scenario sweep")
        
        for i, value in enumerate(values):
            logger.info(f"Running scenario with {scenario_type}={value}")
            
            # Create overrides for this scenario
            overrides = [f"system_model.{scenario_type.lower()}={value}"]
            
            # Add model path override if available
            if model_paths and i < len(model_paths):
                model_path = model_paths[i]
                overrides.append(f"simulation.model_path={model_path}")
                logger.info(f"Using model path for {scenario_type}={value}: {model_path}")
            else:
                # Ensure model_path is set to null if no model paths provided
                overrides.append("simulation.model_path=null")
            
            # Create a modified configuration for this scenario
            from config.loader import apply_overrides
            modified_config = apply_overrides(self.config, overrides)
            
            # Update components for this sweep value (important for system_model and model recreation)
            from config_handler import update_components_for_sweep
            updated_components = update_components_for_sweep(
                components=self.components,
                config=modified_config,
                sweep_param=scenario_type,
                sweep_value=value
            )
            
            # Create a new simulation with the modified config and updated components
            simulation = Simulation(
                config=modified_config,
                components=updated_components,  # Use updated components
                output_dir=self.output_dir / f"{scenario_type}_{value}"
            )
            
            # Run the simulation and store results
            if full_mode:
                result = simulation.run()  # Run complete pipeline
            elif self.config.simulation.evaluate_model and not self.config.simulation.train_model:
                # If in evaluation-only mode, run evaluation instead of training
                logger.info(f"Running evaluation for {scenario_type}={value}")
                result = simulation.run_evaluation()
            elif self.config.simulation.load_model and not self.config.simulation.train_model:
                # If in online learning mode, run online learning
                logger.info(f"Running online learning for {scenario_type}={value}")
                result = simulation.execute_online_learning()
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
        
        # Handle wildcard expansion for model paths
        if '*' in str(model_path):
            import glob
            expanded_paths = glob.glob(str(model_path))
            if not expanded_paths:
                return False, f"No files found matching pattern: {model_path}"
            elif len(expanded_paths) > 1:
                logger.warning(f"Multiple files found for pattern {model_path}: {expanded_paths}. Using the first one.")
            model_path = Path(expanded_paths[0])
            logger.info(f"Expanded wildcard to: {model_path}")
        
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















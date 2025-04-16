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
        sets up dataloaders for training and evaluation.
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
        
        # Create dataloaders
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
        """Execute the evaluation pipeline."""
        logger.info("Starting evaluation pipeline")
        # Placeholder for evaluation implementation
        
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
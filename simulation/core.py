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
from .runners.training import Trainer
from .runners.evaluation import Evaluator

logger = logging.getLogger(__name__)

class Simulation:
    """
    Main simulation controller.
    
    Handles the orchestration of data preparation, model training,
    and evaluation for both single runs and parametric scenarios.
    """
    def __init__(self, config: Config, components: Dict[str, Any], output_dir: Optional[Path] = None):
        self.config = config
        self.components = components
        self.output_dir = output_dir or Path("experiments/results")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize handlers from components
        self.system_model = components.get("system_model")
        self.model = components.get("model")
        self.trajectory_handler = components.get("trajectory_handler")
        
        # Initialize result containers
        self.dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.results = {}
        
    def run(self) -> Dict[str, Any]:
        """
        Run a single simulation.
        
        Returns:
            Dict containing evaluation results.
        """
        logger.info("Starting simulation")
        
        # Execute data pipeline
        self._run_data_pipeline()
        
        # Execute training pipeline if enabled
        if self.config.training.enabled and self.train_dataloader:
            self._run_training_pipeline()
            
        # Execute evaluation pipeline if enabled
        if self.config.simulation.evaluate_model:
            self._run_evaluation_pipeline()
            
        # Save results
        self._save_results()
            
        return self.results
    
    def _run_data_pipeline(self) -> None:
        """
        Execute the data preparation pipeline.
        
        Creates or loads datasets based on configuration and
        sets up dataloaders for training and evaluation.
        """
        logger.info("Starting data pipeline")
        
        # Check if trajectory mode is enabled
        if self.config.trajectory.enabled and self.trajectory_handler:
            logger.info("Using trajectory-based data pipeline")
            dataset, _ = self._create_trajectory_dataset()
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
        
        self.dataset = dataset
        
        # Create dataloaders
        if dataset:
            logger.info("Creating dataloaders")
            self.train_dataloader, self.valid_dataloader = dataset.get_dataloaders(
                batch_size=self.config.training.batch_size
            )
            logger.info(f"Created train dataloader with {len(self.train_dataloader)} batches")
            logger.info(f"Created validation dataloader with {len(self.valid_dataloader)} batches")
        else:
            logger.error("Failed to create or load dataset")
            
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
        """Execute the training pipeline."""
        logger.info("Starting training pipeline")
        
        # Create trainer if not already provided
        trainer = self.components.get("trainer")
        if not trainer:
            # Create trainer directly
            from DCD_MUSIC.src.training import Trainer, TrainingParamsNew
            
            training_params = TrainingParamsNew(
                epochs=self.config.training.epochs,
                batch_size=self.config.training.batch_size,
                optimizer=self.config.training.optimizer,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                training_objective=self.config.training.training_objective
            )
            
            trainer = Trainer(
                model=self.model,
                training_params=training_params,
                show_plots=False
            )
        
        # Train the model
        self.trained_model = trainer.train(
            self.train_dataloader,
            self.valid_dataloader,
            use_wandb=self.config.training.use_wandb,
            save_final=self.config.simulation.save_model,
            load_model=self.config.simulation.load_model
        )
        
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
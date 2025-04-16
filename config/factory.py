"""
Factory module for creating components from configurations.

This module provides functions for creating DCD-MUSIC components from 
configuration objects defined in the schema module.
"""

import sys
import os
import importlib
from typing import Dict, Any, Optional
import warnings
from pathlib import Path
import logging

from config.schema import Config

# Add the DCD_MUSIC module to the path
sys.path.append('./DCD_MUSIC')
# Also add the DCD_MUSIC/src directory to handle absolute imports starting with 'src.'
sys.path.append('./DCD_MUSIC/src')

# Set up logging
logger = logging.getLogger("SubspaceNet.factory")

def _import_from_dcd_music(module_path: str, class_name: str) -> Any:
    """
    Import a class from the DCD_MUSIC package.
    
    Args:
        module_path: The path to the module within the DCD_MUSIC package
        class_name: The name of the class to import
        
    Returns:
        The imported class
    """
    try:
        # Import the module
        module = importlib.import_module(f"DCD_MUSIC.{module_path}")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import {class_name} from DCD_MUSIC.{module_path}: {e}")
        raise ImportError(f"Failed to import {class_name} from DCD_MUSIC.{module_path}")

def _create_system_model_params(config: Config) -> Any:
    """
    Create and configure SystemModelParams from the configuration.
    
    This helper function isolates the SystemModelParams creation to handle
    potential import errors separately.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured SystemModelParams instance
    """
    # Try to import SystemModelParams
    SystemModelParams = _import_from_dcd_music("src.system_model", "SystemModelParams")
    
    # Create and configure system model parameters
    system_model_params = SystemModelParams()
    
    # Make sure wavelength is properly set if not in the config
    config_dict = config.system_model.dict()
    if 'wavelength' not in config_dict:
        config_dict['wavelength'] = 1.0
    
    # Set parameters from configuration
    for key, value in config_dict.items():
        system_model_params.set_parameter(key, value)
        
    return system_model_params

def create_system_model(config: Config) -> Any:
    """
    Create a system model from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        A SystemModel instance
    """
    # Get SystemModelParams
    system_model_params = _create_system_model_params(config)
    
    # Import SystemModel
    SystemModel = _import_from_dcd_music("src.system_model", "SystemModel")
    
    # Create system model
    system_model = SystemModel(system_model_params, nominal=True)
    
    return system_model


def create_dataset(config: Config, system_model: Any) -> Any:
    """
    Create a dataset from configuration.
    
    Args:
        config: Configuration object
        system_model: The system model
        
    Returns:
        A dataset instance
    """
    try:
        # Import TimeSeriesDataset from DCD_MUSIC
        TimeSeriesDataset = _import_from_dcd_music("src.data_handler", "TimeSeriesDataset")
        
        # Import Samples class from DCD_MUSIC
        Samples = _import_from_dcd_music("src.signal_creation", "Samples")
        
        # Create a Samples object (needed for TimeSeriesDataset)
        samples_model = Samples(system_model.params)
        
        # Set the dataset path
        # Use a fixed path relative to the workspace
        datasets_path = Path('data/datasets').absolute()
        # Create the directory if it doesn't exist
        os.makedirs(datasets_path, exist_ok=True)
        
        # Create dataset using the create_dataset function from DCD_MUSIC
        create_dataset_func = _import_from_dcd_music("src.data_handler", "create_dataset")
        dataset, _ = create_dataset_func(
            samples_model=samples_model,
            samples_size=config.dataset.samples_size,
            save_datasets=config.dataset.save_dataset,
            datasets_path=datasets_path,
            true_doa=config.dataset.true_doa_train,
            true_range=config.dataset.true_range_train,
            phase="train"
        )
        
        return dataset
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return None


def create_model(config: Config, system_model: Any) -> Any:
    """
    Create a model from configuration.
    
    Args:
        config: Configuration object
        system_model: The system model
        
    Returns:
        A model instance
    """
    model_type = config.model.type
    model_params = config.model.params.dict()

    if model_type == "SubspaceNet":
        SubspaceNet = _import_from_dcd_music("src.models_pack.subspacenet", "SubspaceNet")
        
        # Extract parameters
        tau = model_params.get("tau", 8)
        diff_method = model_params.get("diff_method", "music_2D")
        train_loss_type = model_params.get("train_loss_type", "music_spectrum")
        field_type = model_params.get("field_type", "Near")
        regularization = model_params.get("regularization")
        variant = model_params.get("variant", "small")
        norm_layer = model_params.get("norm_layer", False)
        batch_norm = model_params.get("batch_norm", False)
        
        # Validate regularization parameter and convert "null" to None
        valid_regularization_values = [None, "aic", "mdl", "threshold", "null"]
        if regularization not in valid_regularization_values:
            raise ValueError(f"Invalid regularization value: {regularization}. " 
                            f"Must be one of {valid_regularization_values}")
        
        # Convert "null" to None for proper handling
        if regularization == "null":
            regularization = None
        
        # Create model
        model = SubspaceNet(
            tau=tau,
            diff_method=diff_method,
            train_loss_type=train_loss_type,
            system_model=system_model,
            field_type=field_type,
            regularization=regularization,
            variant=variant,
            norm_layer=norm_layer,
            batch_norm=batch_norm
        )
        
        return model
        
    elif model_type == "DCD-MUSIC":
        DCDMUSIC = _import_from_dcd_music("src.models_pack.dcd_music", "DCDMUSIC")
        
        # Extract parameters
        tau = model_params.get("tau", 8)
        diff_method = model_params.get("diff_method", ("esprit", "music_1d"))
        train_loss_type = model_params.get("train_loss_type", ("rmspe", "rmspe"))
        regularization = model_params.get("regularization")
        variant = model_params.get("variant", "small")
        norm_layer = model_params.get("norm_layer", False)
        
        # Create model
        model = DCDMUSIC(
            tau=tau,
            system_model=system_model,
            diff_method=diff_method,
            train_loss_type=train_loss_type,
            regularization=regularization,
            variant=variant,
            norm_layer=norm_layer
        )
        
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_trainer(config: Config, model: Any, dataset: Any) -> Any:
    """
    Create a trainer from configuration.
    
    Args:
        config: Configuration object
        model: The model to train
        dataset: The dataset to use for training
        
    Returns:
        A trainer instance or None if creation fails
    """
    try:
        # Import Trainer class
        Trainer = _import_from_dcd_music("src.training", "Trainer")
        
        # Import TrainingParamsNew
        TrainingParamsNew = _import_from_dcd_music("src.training", "TrainingParamsNew")
        
        # Extract training parameters
        training_params = TrainingParamsNew(
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            optimizer=config.training.optimizer,
            scheduler=config.training.scheduler,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            step_size=config.training.step_size,
            gamma=config.training.gamma,
            training_objective=config.training.training_objective,
            use_wandb=config.training.use_wandb,
            simulation_name=config.training.simulation_name
        )
        
        # Create trainer 
        # Note: The Trainer class doesn't accept a dataset in its constructor
        # The dataset is passed later to the train method
        trainer = Trainer(
            model=model,
            training_params=training_params,
            show_plots=False
        )
        
        # Store the dataset reference for later use during run_experiment
        trainer.dataset = dataset
        
        return trainer
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return None


def create_trajectory_data_handler(config: Config, system_model: Any) -> Any:
    """
    Create a trajectory data handler from configuration.
    
    Args:
        config: Configuration object
        system_model: The system model
        
    Returns:
        A TrajectoryDataHandler instance
    """
    try:
        # Import the TrajectoryDataHandler class
        from simulation.runners.data import TrajectoryDataHandler
        
        # Create the data handler with system model
        handler = TrajectoryDataHandler(
            system_model_params=system_model.params,
            config=config
        )
        
        return handler
    except Exception as e:
        logger.error(f"Failed to create trajectory data handler: {e}")
        return None


def create_components_from_config(config: Config) -> Dict[str, Any]:
    """
    Create all components from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        A dictionary containing all created components
    """
    logger = logging.getLogger("SubspaceNet")
    components = {}
    
    try:
        # Create system model
        logger.info("Creating system model...")
        system_model = create_system_model(config)
        if system_model is None:
            raise ValueError("Failed to create system model")
        components["system_model"] = system_model
        
        # Create trajectory data handler if enabled
        if config.trajectory.enabled:
            logger.info("Creating trajectory data handler...")
            trajectory_handler = create_trajectory_data_handler(config, system_model)
            if trajectory_handler is not None:
                components["trajectory_handler"] = trajectory_handler
                logger.info("Trajectory data handler created successfully")
            else:
                logger.warning("Trajectory data handler creation failed")
        
        # Create dataset if create_data is True
        if config.dataset.create_data:
            logger.info("Creating dataset...")
            dataset = create_dataset(config, system_model)
            if dataset is not None:
                components["dataset"] = dataset
                logger.info("Dataset created successfully")
            else:
                logger.warning("Dataset creation failed, will continue with other components")
        else:
            logger.info("Skipping dataset creation (create_data is False)")
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config, system_model)
        if model is None:
            raise ValueError("Failed to create model")
        components["model"] = model
        logger.info("Model created successfully")
        
        # Create trainer if training is enabled and we have a dataset
        if config.training.enabled and "dataset" in components:
            logger.info("Creating trainer...")
            trainer = create_trainer(config, model, components["dataset"])
            if trainer is not None:
                components["trainer"] = trainer
                logger.info("Trainer created successfully")
            else:
                logger.warning("Trainer creation failed, but other components are available")
        elif config.training.enabled:
            logger.warning("Training is enabled but dataset is not available; skipping trainer creation")
        else:
            logger.info("Training is disabled; skipping trainer creation")
        
        return components
    except Exception as e:
        logger.error(f"Failed to create components from config: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "config": config
        } 
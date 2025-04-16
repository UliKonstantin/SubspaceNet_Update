"""
SubspaceNet experiment runner module.

This module contains functions for running experiments with SubspaceNet.
"""

import os
import sys
import json
import logging
import time
import importlib
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger('SubspaceNet.experiments')

def run_experiment(config: Any, components: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run an experiment with the given configuration and components.
    
    Args:
        config: The configuration object
        components: Dictionary of components created from the configuration
        output_dir: Optional output directory for experiment results
        
    Returns:
        Dictionary of experiment results
    """
    logger.info("Starting experiment")
    
    # Create output directory if specified
    out_path = None
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment results will be saved to {out_path}")
    
    # Extract components
    system_model = components.get("system_model")
    dataset = components.get("dataset")
    model = components.get("model")
    trainer = components.get("trainer")
    
    # Check if we have the minimum required components
    if not system_model:
        logger.error("Cannot run experiment: System model not available")
        return {"error": "System model not available"}
    
    # Log component information
    if system_model:
        logger.info(f"System model: N={system_model.params.N}, M={system_model.params.M}, field_type={system_model.params.field_type}")
    
    if dataset:
        try:
            logger.info(f"Dataset: {len(dataset)} samples")
        except:
            logger.info("Dataset available but could not determine size")
    else:
        logger.warning("Dataset not available")
    
    if model:
        try:
            logger.info(f"Model: {model._get_name()}")
        except:
            logger.info("Model available but could not determine name")
    else:
        logger.warning("Model not available")
        
    if trainer:
        logger.info(f"Trainer configured with {config.training.epochs} epochs")
    else:
        logger.warning("Trainer not available")
    
    # Run training if enabled and prerequisites are available
    results = {}
    if config.simulation.train_model and trainer and dataset and model:
        logger.info("Training model...")
        
        try:
            # Create data loaders
            # In a real implementation, we'd use the actual train/test split
            # For this test, we'll just use a simple random split
            train_size = int(len(dataset) * 0.8)
            test_size = len(dataset) - train_size
            
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.training.batch_size, 
                shuffle=True
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config.training.batch_size, 
                shuffle=False
            )
            
            logger.info(f"Created data loaders with {train_size} training samples and {test_size} test samples")
            
            # Perform actual training
            logger.info("Starting model training...")
            start_time = time.time()
            
            try:
                # Call the trainer's train method with the data loaders
                try:
                    train_history = trainer.train(
                        train_dataloader=train_loader,
                        valid_dataloader=test_loader,
                        use_wandb=config.training.use_wandb,
                        save_final=config.simulation.save_model,
                        load_model=config.simulation.load_model
                    )
                    
                    training_time = time.time() - start_time
                    logger.info(f"Training completed in {training_time:.2f} seconds")
                    
                    results["training_completed"] = True
                    results["training_time"] = training_time
                    
                except NotImplementedError as nie:
                    # Handle the case where training is not implemented for this model
                    logger.warning(f"Training not implemented: {nie}")
                    results["training_skipped"] = "Not implemented for this model"
                
                results["train_size"] = train_size
                results["test_size"] = test_size
                
                # Save model if configured
                if config.simulation.save_model and out_path:
                    model_path = out_path / "trained_model.pth"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Model saved to {model_path}")
                    results["model_saved"] = str(model_path)
                
            except Exception as e:
                logger.error(f"Error during model training: {e}")
                results["training_error"] = str(e)
            
        except Exception as e:
            logger.error(f"Error during training setup: {e}")
            results["training_error"] = str(e)
            
    elif config.simulation.train_model:
        logger.warning("Training is enabled but prerequisites are missing")
        results["training_skipped"] = "Missing prerequisites"
    
    # Run evaluation if enabled and prerequisites are available
    if config.simulation.evaluate_model and model:
        logger.info("Evaluating model...")
        try:
            # Create test data loader if not already created
            if 'test_loader' not in locals() and dataset:
                test_size = int(len(dataset) * 0.2)
                _, test_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=config.training.batch_size, 
                    shuffle=False
                )
                logger.info(f"Created evaluation data loader with {test_size} samples")
            
            if 'test_loader' in locals():
                # Import evaluation function
                evaluate_dnn_model = _import_from_dcd_music("src.evaluation", "evaluate_dnn_model")
                
                # Run evaluation
                logger.info("Running model evaluation...")
                evaluation_results = evaluate_dnn_model(
                    model=model,
                    dataset=test_loader,  # Use 'dataset' instead of 'test_dataloader'
                    mode="test"  # Specify 'test' mode
                )
                
                results["evaluation_completed"] = True
                results["evaluation_results"] = evaluation_results
                
                logger.info(f"Evaluation completed")
                
                # Plot results if configured
                if config.simulation.plot_results and out_path:
                    logger.info("Plotting would be done here in a real implementation")
                    results["plots_generated"] = False
            else:
                logger.warning("No test dataset available for evaluation")
                results["evaluation_skipped"] = "No test dataset"
        
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            results["evaluation_error"] = str(e)
            
    elif config.simulation.evaluate_model:
        logger.warning("Evaluation is enabled but model is not available")
        results["evaluation_skipped"] = "Model not available"
    
    # Save results if output directory is specified
    if out_path:
        # Save configuration
        config_path = out_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.dict(), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Save experiment results
        results_path = out_path / "results.json"
        with open(results_path, "w") as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {results_path}")
        
        # Save model summary if available
        if model:
            model_info_path = out_path / "model_info.txt"
            with open(model_info_path, "w") as f:
                f.write(f"Model: {model._get_name()}\n")
                f.write(f"Model type: {type(model).__name__}\n")
                try:
                    # Try to get model parameters count
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    f.write(f"Total parameters: {total_params}\n")
                    f.write(f"Trainable parameters: {trainable_params}\n")
                except:
                    f.write("Could not determine model parameters\n")
            logger.info(f"Model information saved to {model_info_path}")
    
    logger.info("Experiment completed successfully")
    return results

def _import_from_dcd_music(module_path: str, class_name: str) -> Any:
    """Helper function to import from DCD_MUSIC package"""
    try:
        # Import the module
        module = importlib.import_module(f"DCD_MUSIC.{module_path}")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import {class_name} from DCD_MUSIC.{module_path}: {e}")
        return None 
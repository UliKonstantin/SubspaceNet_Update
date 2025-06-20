#!/usr/bin/env python3
"""
SubspaceNet - Command Line Interface

This is the main entry point for the SubspaceNet configuration framework.
"""

import sys
import click
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from config.loader import save_config
from config_handler import setup_configuration
from experiments.runner import run_experiment
from cli.commands import show_command, save_command
from cli.options import config_option, output_option, override_option
from simulation.core import Simulation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SubspaceNet')

@click.group()
def cli():
    """SubspaceNet - A framework for direction-of-arrival estimation."""
    # Add DCD_MUSIC to Python path if it exists
    dcd_music_path = Path(__file__).parent / "DCD_MUSIC"
    if dcd_music_path.exists():
        sys.path.append(str(dcd_music_path.parent))

@cli.command('run')
@config_option
@output_option
@override_option
@click.option('--scenario', '-s', type=click.Choice(['training', 'evaluation', 'online_learning', 'full']), default='training',
              help='Scenario to run (training, evaluation, online_learning, or full for all components)')
def run_command(config: str, output: Optional[str], override: List[str], scenario: str):
    """Run an experiment with the specified configuration."""
    try:
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Create simulation instance
        logger.info(f"Running {scenario} scenario")
        sim = Simulation(config_obj, components, output_dir)
        
        # Run the appropriate scenario
        if scenario == 'training':
            results = sim.run_training()
        elif scenario == 'evaluation':
            results = sim.run_evaluation()
        elif scenario == 'online_learning':
            results = sim.run_online_learning()
        elif scenario == 'full':
            results = sim.run()  # Run the complete pipeline with all components
        
        logger.info(f"{scenario.capitalize()} completed successfully")
        
    except Exception as e:
        logger.error(f"Error running {scenario}: {e}", exc_info=True)
        sys.exit(1)

@cli.command('evaluate')
@config_option
@output_option
@override_option
@click.option('--model', '-m', type=str, required=True, help='Path to the trained model to evaluate')
@click.option('--trajectory/--no-trajectory', default=False, help='Enable trajectory-based data')
@click.option('--scenario', '-s', type=str, help='Parameter to sweep during evaluation (SNR, M, eta, etc.)')
@click.option('--values', '-v', multiple=True, type=float, help='Values for the scenario parameter (can be used multiple times)')
def evaluate_command(config: str, output: Optional[str], override: List[str],
                    model: str, trajectory: bool, scenario: Optional[str], values: List[float]):
    """Evaluate a pre-trained model without training. Optionally run parameter sweeps."""
    try:
        # Add required overrides to skip training and use the provided model
        override = list(override) + [
            "simulation.train_model=false",
            "simulation.load_model=true",
            f"simulation.model_path={model}",
            "simulation.evaluate_model=true"
        ]
        
        # Add trajectory override if enabled
        if trajectory:
            override.append("trajectory.enabled=true")
            
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Create simulation
        logger.info(f"Evaluating model: {model}")
        sim = Simulation(config_obj, components, output_dir)
        
        # Check if we need to run a parameter sweep
        if scenario:
            # Try multiple ways to get sweep values if not provided via command line
            config_values = None
            
            # Try the standard pydantic model way
            if not values and hasattr(config_obj, 'evaluation'):
                if hasattr(config_obj.evaluation, 'sweep_values'):
                    config_values = config_obj.evaluation.sweep_values
                    logger.info(f"Using sweep_values from config: {config_values}")
                
                # Try dictionary access (sometimes needed for custom attributes)
                elif hasattr(config_obj.evaluation, '__dict__'):
                    eval_dict = config_obj.evaluation.__dict__
                    if 'sweep_values' in eval_dict:
                        config_values = eval_dict['sweep_values']
                        logger.info(f"Using sweep_values from __dict__: {config_values}")
            
            # If scenario is "eta", use default calibration error values if not provided
            if not values and not config_values and scenario.lower() == "eta":
                config_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
                logger.info(f"Using default eta values: {config_values}")
                
            # If scenario is "snr", use default SNR values if not provided
            elif not values and not config_values and scenario.lower() == "snr":
                config_values = [-10, -5, 0, 5, 10, 15, 20]
                logger.info(f"Using default SNR values: {config_values}")
                
            # If scenario is "M", use default source count values if not provided
            elif not values and not config_values and scenario.lower() == "m":
                config_values = [1, 2, 3, 4, 5]
                logger.info(f"Using default source count (M) values: {config_values}")
                
            # Use config values if available, otherwise use command line values
            if config_values:
                values = config_values
            
            # Make sure we have values to sweep over
            if not values:
                logger.error(f"Scenario '{scenario}' specified but no values provided. Either use -v option or add sweep_values to config.")
                sys.exit(1)
                
            logger.info(f"Running evaluation sweep on {scenario} with values {values}")
            
            # For each value, run evaluation
            scenario_results = {}
            for value in values:
                logger.info(f"Evaluating with {scenario}={value}")
                
                # Determine the correct config section for the scenario parameter
                if scenario.lower() in ['trajectory_length']:
                    override_path = f"trajectory.{scenario.lower()}={value}"
                elif scenario.lower() in ['snr', 'm', 'n', 't', 'eta', 'bias', 'sv_noise_var']:
                    override_path = f"system_model.{scenario.lower()}={value}"
                else:
                    # Default to system_model for backward compatibility
                    override_path = f"system_model.{scenario.lower()}={value}"
                
                # Create a modified configuration for this scenario value
                from config.loader import apply_overrides
                modified_config = apply_overrides(
                    config_obj,
                    [override_path]
                )
                
                # Update components for this sweep value
                from config_handler import update_components_for_sweep
                updated_components = update_components_for_sweep(
                    components=components,
                    config=modified_config,
                    sweep_param=scenario,
                    sweep_value=value
                )
                
                # Create a new simulation with the modified config and updated components
                scenario_sim = Simulation(
                    config=modified_config,
                    components=updated_components,
                    output_dir=output_dir / f"{scenario}_{value}"
                )
                
                # Run evaluation with this configuration
                result = scenario_sim.run_evaluation()
                scenario_results[value] = result
            
            # Store and log results
            sim.results[scenario] = scenario_results
            logger.info(f"Evaluation sweep completed with {len(scenario_results)} results")

            # --- Plotting loss vs. swept parameter ---
            try:
                from utils.plotting import plot_loss_vs_scenario
                plot_path = plot_loss_vs_scenario(scenario_results, scenario, output_dir)
                logger.info(f"Saved loss plot to {plot_path}")
            except Exception as e:
                logger.warning(f"Could not plot loss vs. {scenario}: {e}")
            
        else:
            # Run standard evaluation
            results = sim.run_evaluation()
            
            if results.get("status") == "error":
                logger.error(f"Evaluation failed: {results.get('message')}")
                sys.exit(1)
                
            logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)
        sys.exit(1)

@cli.command('simulate')
@config_option
@output_option
@override_option
@click.option('--scenario', '-s', type=str, help='Run scenario (SNR, T, M, eta, etc.)')
@click.option('--values', '-v', multiple=True, type=float, help='Values to test in scenario (can be used multiple times)')
@click.option('--trajectory/--no-trajectory', default=False, help='Enable trajectory-based data')
@click.option('--mode', type=click.Choice(['training', 'full']), default='training',
              help='Simulation mode: training only or full pipeline (training, evaluation, online_learning)')
def simulate_command(config: str, output: Optional[str], override: List[str], 
                    scenario: Optional[str], values: List[float], trajectory: bool,
                    mode: str):
    """Run a simulation with the specified configuration."""
    try:
        # Add trajectory override if enabled
        if trajectory:
            override = list(override) + ["trajectory.enabled=true"]
            
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Create simulation
        logger.info("Creating simulation")
        sim = Simulation(config_obj, components, output_dir)
        
        # Check if scenario and values are provided via command line
        # If not, try to read them from the config file
        if not scenario and hasattr(config_obj, 'scenario_config') and config_obj.scenario_config:
            scenario = config_obj.scenario_config.type
            logger.info(f"Using scenario type from config: {scenario}")
            
        if not values and hasattr(config_obj, 'scenario_config') and config_obj.scenario_config and config_obj.scenario_config.values:
            values = config_obj.scenario_config.values
            logger.info(f"Using scenario values from config: {values}")
        
        # Run a single simulation or a scenario
        if scenario and values:
            logger.info(f"Running {scenario} scenario with values {values}")
            results = sim.run_scenario(scenario, list(values), full_mode=(mode=='full'))
            logger.info(f"Scenario completed with {len(results)} results")
        else:
            if mode == 'full':
                logger.info("Running full simulation (training, evaluation, and online learning)")
                results = sim.run()
            else:
                logger.info("Running training-only simulation")
                results = sim.run_training()
            logger.info(f"{mode.capitalize()} simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        sys.exit(1)

@cli.command('online_learning')
@config_option
@output_option
@override_option
@click.option('--model', '-m', type=str, required=True, help='Path to the trained model for online learning')
def online_learning_command(config: str, output: Optional[str], override: List[str], model: str):
    """Run online learning with a pre-trained model."""
    try:
        # Add required overrides
        override = list(override) + [
            "simulation.train_model=false",
            "simulation.load_model=true",
            f"simulation.model_path={model}",
            "online_learning.enabled=true"
        ]
        
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Create simulation
        logger.info(f"Running online learning with model: {model}")
        sim = Simulation(config_obj, components, output_dir)
        
        # Run online learning scenario
        results = sim.run_online_learning()
        
        if results.get("status") == "error":
            logger.error(f"Online learning failed: {results.get('message')}")
            sys.exit(1)
            
        logger.info("Online learning completed successfully")
        
    except Exception as e:
        logger.error(f"Error running online learning: {e}", exc_info=True)
        sys.exit(1)

# Add other commands
cli.add_command(show_command)
cli.add_command(save_command)
cli.add_command(online_learning_command)

if __name__ == '__main__':
    cli() 
   
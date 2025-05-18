#!/usr/bin/env python3
"""
SubspaceNet - Command Line Interface

This is the main entry point for the SubspaceNet configuration framework.
"""

import sys
import click
import logging
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
def run_command(config: str, output: Optional[str], override: List[str]):
    """Run an experiment with the specified configuration."""
    try:
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Run experiment
        logger.info("Running experiment")
        results = run_experiment(config_obj, components, output_dir)
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        sys.exit(1)

@cli.command('evaluate')
@config_option
@output_option
@override_option
@click.option('--model', '-m', type=str, required=True, help='Path to the trained model to evaluate')
@click.option('--trajectory/--no-trajectory', default=False, help='Enable trajectory-based data')
def evaluate_command(config: str, output: Optional[str], override: List[str],
                    model: str, trajectory: bool):
    """Evaluate a pre-trained model without training."""
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
        
        # Run simulation (will skip training and load the model)
        results = sim.run()
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)
        sys.exit(1)

@cli.command('simulate')
@config_option
@output_option
@override_option
@click.option('--scenario', '-s', type=str, help='Run scenario (SNR, T, M, etc.)')
@click.option('--values', '-v', multiple=True, type=float, help='Values to test in scenario (can be used multiple times)')
@click.option('--trajectory/--no-trajectory', default=False, help='Enable trajectory-based data')
def simulate_command(config: str, output: Optional[str], override: List[str], 
                    scenario: Optional[str], values: List[float], trajectory: bool):
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
        
        # Run a single simulation or a scenario
        if scenario and values:
            logger.info(f"Running {scenario} scenario with values {values}")
            results = sim.run_scenario(scenario, list(values))
            logger.info(f"Scenario completed with {len(results)} results")
        else:
            logger.info("Running single simulation")
            results = sim.run()
            logger.info("Simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        sys.exit(1)

# Add other commands
cli.add_command(show_command)
cli.add_command(save_command)

if __name__ == '__main__':
    cli() 
   
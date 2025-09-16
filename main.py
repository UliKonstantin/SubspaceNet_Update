#!/usr/bin/env python3
#Train command over multiple snrs python3 main.py simulate -c configs/training_config/Random_basemodel_training_config.yaml -o experiments/results/my_snr_experiment -s snr -v -10 -v -5 -v 0 v 5 -v 10 --mode training
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
import torch

from config.loader import save_config
from config_handler import setup_configuration
from experiments.runner import run_experiment
from cli.commands import show_command, save_command
from cli.options import config_option, output_option, override_option
from simulation.core import Simulation
from utils.logging_utils import setup_logging_from_config
from utils.plotting import plot_scenario_results

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    logger = logging.getLogger('SubspaceNet')  # Initialize logger early
    try:
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Setup logging from config
        setup_logging_from_config(config_obj.logging, output_dir)
        
        # Create simulation instance
        logger.info(f"Running {scenario} scenario")
        sim = Simulation(config_obj, components, output_dir)
        
        # Run the appropriate scenario
        if scenario == 'training':
            results = sim.run_training()
        elif scenario == 'evaluation':
            results = sim.run_evaluation()
        elif scenario == 'online_learning':
            results = sim.execute_online_learning()
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
@click.option('--scenario', '-s', type=str, help='Parameter to sweep during evaluation (SNR, M, eta, kalman_noise, etc.)')
@click.option('--values', '-v', multiple=True, type=float, help='Values for the scenario parameter (can be used multiple times)')
def evaluate_command(config: str, output: Optional[str], override: List[str],
                    model: str, trajectory: bool, scenario: Optional[str], values: List[float]):
    """Evaluate a pre-trained model without training. Optionally run parameter sweeps."""
    # Initialize logger at function start to avoid UnboundLocalError
    logger = None
    
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
        
        # Setup logging from config
        setup_logging_from_config(config_obj.logging, output_dir)
        logger = logging.getLogger('SubspaceNet')
        
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
                
            # If scenario is "kalman_noise", handle measurement and process noise sweep
            elif not values and not config_values and scenario.lower() == "kalman_noise":
                # This is a special 2D sweep scenario - values will be handled differently
                config_values = None
                logger.info("Using Kalman noise sweep - will use default measurement and process noise ranges")
                
            # Use config values if available, otherwise use command line values
            if config_values:
                values = config_values
            
            # Handle special 2D sweep scenario for Kalman noise
            if scenario.lower() == "kalman_noise":
                logger.info("Running 2D evaluation sweep on Kalman filter noise parameters")
                
                # Define default ranges for measurement and process noise std dev
                measurement_noise_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5] if not values else values[:len(values)//2] if len(values) > 1 else [0.001, 0.01, 0.1]
                process_noise_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5] if not values else values[len(values)//2:] if len(values) > 1 else [0.001, 0.01, 0.1]
                
                logger.info(f"Measurement noise std dev values: {measurement_noise_values}")
                logger.info(f"Process noise std dev values: {process_noise_values}")
                
                # 2D sweep over all combinations
                scenario_results = {}
                total_combinations = len(measurement_noise_values) * len(process_noise_values)
                combination_count = 0
                
                for meas_noise in measurement_noise_values:
                    scenario_results[meas_noise] = {}
                    for proc_noise in process_noise_values:
                        combination_count += 1
                        logger.info(f"Evaluating combination {combination_count}/{total_combinations}: measurement_noise={meas_noise}, process_noise={proc_noise}")
                        
                        # Create overrides for both noise parameters
                        overrides = [
                            f"kalman_filter.measurement_noise_std_dev={meas_noise}",
                            f"kalman_filter.process_noise_std_dev={proc_noise}"
                        ]
                        
                        # Create a modified configuration for this parameter combination
                        from config.loader import apply_overrides
                        modified_config = apply_overrides(config_obj, overrides)
                        
                        # Update components for this sweep combination
                        from config_handler import update_components_for_sweep
                        updated_components = update_components_for_sweep(
                            components=components,
                            config=modified_config,
                            sweep_param="kalman_noise",
                            sweep_value=(meas_noise, proc_noise)
                        )
                        
                        # Create a new simulation with the modified config and updated components
                        scenario_sim = Simulation(
                            config=modified_config,
                            components=updated_components,
                            output_dir=output_dir / f"kalman_noise_m{meas_noise}_p{proc_noise}"
                        )
                        
                        # Run evaluation with this configuration
                        result = scenario_sim.run_evaluation()
                        scenario_results[meas_noise][proc_noise] = result
                        
                        logger.info(f"Completed combination {combination_count}/{total_combinations}")
                
            else:
                # Make sure we have values to sweep over for single parameter scenarios
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
            
            if scenario.lower() == "kalman_noise":
                total_results = sum(len(inner_dict) for inner_dict in scenario_results.values())
                logger.info(f"Kalman noise sweep completed with {total_results} combinations across {len(scenario_results)} measurement noise values")
                
                # Log summary of results
                logger.info("=== KALMAN NOISE SWEEP RESULTS SUMMARY ===")
                for meas_noise in sorted(scenario_results.keys()):
                    for proc_noise in sorted(scenario_results[meas_noise].keys()):
                        result = scenario_results[meas_noise][proc_noise]
                        if 'dnn_loss' in result:
                            logger.info(f"  meas_noise={meas_noise:6.3f}, proc_noise={proc_noise:6.3f} => DNN Loss: {result['dnn_loss']:.6f}")
                        else:
                            logger.info(f"  meas_noise={meas_noise:6.3f}, proc_noise={proc_noise:6.3f} => No DNN loss recorded")
                
                # --- Plotting 2D heatmap for Kalman noise sweep ---
                try:
                    from utils.plotting import plot_2d_kalman_noise_sweep
                    plot_path = plot_2d_kalman_noise_sweep(scenario_results, output_dir)
                    logger.info(f"Saved 2D Kalman noise heatmap to {plot_path}")
                except Exception as e:
                    logger.warning(f"Could not plot 2D Kalman noise heatmap: {e}")
                    
            else:
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
        if logger is not None:
            logger.error(f"Error evaluating model: {e}", exc_info=True)
        else:
            # Fallback logging if logger is not available
            print(f"Error evaluating model: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command('simulate')
@config_option
@output_option
@override_option
@click.option('--scenario', '-s', type=str, help='Run scenario (SNR, T, M, eta, etc.)')
@click.option('--values', '-v', multiple=True, type=float, help='Values to test in scenario (can be used multiple times)')
@click.option('--trajectory/--no-trajectory', default=False, help='Enable trajectory-based data')
@click.option('--mode', type=click.Choice(['training', 'full', 'online_learning']), default='training',
              help='Simulation mode: training only, full pipeline (training, evaluation, online_learning), or online_learning only')
def simulate_command(config: str, output: Optional[str], override: List[str], 
                    scenario: Optional[str], values: List[float], trajectory: bool,
                    mode: str):
    """Run a simulation with the specified configuration."""
    # Initialize logger at function start to avoid UnboundLocalError
    logger = None
    
    try:
        # Add trajectory override if enabled
        if trajectory:
            override = list(override) + ["trajectory.enabled=true"]
            
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Setup logging from config
        setup_logging_from_config(config_obj.logging, output_dir)
        logger = logging.getLogger('SubspaceNet')
        
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
            
            # Plot scenario results if it's an SNR scenario
            if scenario.lower() == 'snr' and mode == 'online_learning':
                from utils.plotting import plot_scenario_results, plot_performance_improvement_table
                plot_scenario_results(results, sim.output_dir)
                plot_performance_improvement_table(results, sim.output_dir)
        else:
            if mode == 'full':
                logger.info("Running full simulation (training, evaluation, and online learning)")
                results = sim.run()
            elif mode == 'online_learning':
                logger.info("Running online learning simulation")
                results = sim.execute_online_learning()
            else:
                logger.info("Running training-only simulation")
                results = sim.run_training()
            logger.info(f"{mode.capitalize()} simulation completed successfully")
        
    except Exception as e:
        if logger is not None:
            logger.error(f"Error running simulation: {e}", exc_info=True)
        else:
            # Fallback logging if logger is not available
            print(f"Error running simulation: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command('online_learning')
@config_option
@output_option
@override_option
@click.option('--model', '-m', type=str, required=False, help='Path to the trained model for online learning (overrides config)')
@click.option('--scenario', '-s', type=str, help='Run scenario (4d_grid for eta/process_noise/kalman_noise grid search)')
@click.option('--eta-values', multiple=True, type=float, help='Values for eta.')
@click.option('--process-noise-values', multiple=True, type=float, help='Values for trajectory process noise.')
@click.option('--kf-process-noise-values', multiple=True, type=float, help='Values for Kalman filter process noise std dev.')
@click.option('--kf-measurement-noise-values', multiple=True, type=float, help='Values for Kalman filter measurement noise std dev.')
def online_learning_command(config: str, output: Optional[str], override: List[str], model: str, scenario: Optional[str], 
                            eta_values: List[float], process_noise_values: List[float], 
                            kf_process_noise_values: List[float], kf_measurement_noise_values: List[float]):
    """Run online learning with a pre-trained model."""
    # Initialize logger at function start to avoid UnboundLocalError
    logger = None
    
    try:
        # Use config_handler to set up configuration and components
        config_obj, components, output_dir = setup_configuration(config, output, override)
        
        # Setup logging from config
        setup_logging_from_config(config_obj.logging, output_dir)
        logger = logging.getLogger('SubspaceNet')
        
        # Validate that model path is provided either via flag or config
        if not model and not config_obj.simulation.model_path:
            raise ValueError("Model path must be provided either via --model flag or in config file (simulation.model_path)")
        
        # Apply required overrides for online learning
        from config.loader import apply_overrides
        online_learning_overrides = [
            "simulation.train_model=false",
            "simulation.load_model=true",
            "online_learning.enabled=true"
        ]
        
        # Only override model_path if -m flag was provided
        if model:
            online_learning_overrides.append(f"simulation.model_path={model}")
        
        modified_config = apply_overrides(config_obj, online_learning_overrides)
        
        # Check if we need to run a parameter sweep
        if scenario and scenario.lower() == "4d_grid":
            logger.info("Running 4D grid search on eta, process_noise, kalman_filter process_noise_std_dev, and measurement_noise_std_dev")
            
            # Use provided values or fall back to defaults
            if not eta_values:
                eta_values = [0.0, 0.01, 0.015, 0.02, 0.03]
            if not process_noise_values:
                process_noise_values = [0.001, 0.01, 0.1]
            if not kf_process_noise_values:
                kf_process_noise_values = [0.001, 0.01, 0.1]
            if not kf_measurement_noise_values:
                kf_measurement_noise_values = [0.001, 0.01, 0.1]
            
            logger.info(f"Eta values: {eta_values}")
            logger.info(f"Process noise values: {process_noise_values}")
            logger.info(f"Kalman filter process noise std dev values: {kf_process_noise_values}")
            logger.info(f"Kalman filter measurement noise std dev values: {kf_measurement_noise_values}")
            
            # 4D sweep over all combinations - restructured to run eta last
            scenario_results = {}
            total_combinations = len(eta_values) * len(process_noise_values) * len(kf_process_noise_values) * len(kf_measurement_noise_values)
            combination_count = 0
            
            for proc_noise in process_noise_values:
                scenario_results[proc_noise] = {}
                for kf_proc_noise in kf_process_noise_values:
                    scenario_results[proc_noise][kf_proc_noise] = {}
                    for kf_meas_noise in kf_measurement_noise_values:
                        scenario_results[proc_noise][kf_proc_noise][kf_meas_noise] = {}
                        for eta in eta_values:
                            combination_count += 1
                            logger.info(f"Online learning combination {combination_count}/{total_combinations}: proc_noise={proc_noise}, kf_process_noise={kf_proc_noise}, kf_measurement_noise={kf_meas_noise}, eta={eta}")
                            
                            # Create overrides for all 4 parameters
                            grid_overrides = [
                                f"online_learning.max_eta={eta}",
                                f"online_learning.eta_increment={eta}",
                                f"trajectory.sine_accel_noise_std={proc_noise}",
                                f"trajectory.mult_noise_base_std={proc_noise}",
                                f"trajectory.random_walk_std_dev={proc_noise}",
                                f"kalman_filter.process_noise_std_dev={kf_proc_noise}",
                                f"kalman_filter.measurement_noise_std_dev={kf_meas_noise}"
                            ]
                            
                            # Create a modified configuration for this parameter combination
                            grid_modified_config = apply_overrides(modified_config, grid_overrides)
                            
                            # Update components for this sweep combination
                            from config_handler import update_components_for_sweep
                            updated_components = update_components_for_sweep(
                                components=components,
                                config=grid_modified_config,
                                sweep_param="4d_grid",
                                sweep_value=(proc_noise, kf_proc_noise, kf_meas_noise, eta)
                            )
                            
                            # Create output directory with descriptive name
                            grid_output_dir = output_dir / f"4d_grid_pn{proc_noise}_kf_pn{kf_proc_noise}_kf_mn{kf_meas_noise}_eta{eta}"
                            
                            # Create a new simulation with the modified config and updated components
                            scenario_sim = Simulation(
                                config=grid_modified_config,
                                components=updated_components,
                                output_dir=grid_output_dir
                            )
                            
                            # Run online learning with this configuration
                            result = scenario_sim.execute_online_learning()
                            scenario_results[proc_noise][kf_proc_noise][kf_meas_noise][eta] = result
                            
                            logger.info(f"Completed combination {combination_count}/{total_combinations}")
            
            # Store results and log summary
            total_results = sum(
                sum(sum(len(inner_dict3) for inner_dict3 in inner_dict2.values()) 
                    for inner_dict2 in inner_dict1.values()) 
                for inner_dict1 in scenario_results.values()
            )
            logger.info(f"4D grid search completed with {total_results} combinations")
            
            # Generate eta comparison plots
            try:
                from utils.plotting import plot_eta_comparison_4d_grid
                logger.info("Starting eta comparison plotting...")
                saved_plots = plot_eta_comparison_4d_grid(scenario_results, output_dir)
                logger.info(f"Eta comparison plotting completed: {len(saved_plots)} plots saved")
            except Exception as e:
                logger.error(f"Error during eta comparison plotting: {e}")
                logger.debug("Plotting error details:", exc_info=True)
            
            logger.info("4D grid search completed successfully")
            
        else:
            # Run standard online learning
            sim = Simulation(modified_config, components, output_dir)
            
            # Run online learning scenario
            results = sim.execute_online_learning()
            
            if results.get("status") == "error":
                logger.error(f"Online learning failed: {results.get('message')}")
                sys.exit(1)
                
            logger.info("Online learning completed successfully")
        
    except Exception as e:
        if logger is not None:
            logger.error(f"Error running online learning: {e}", exc_info=True)
        else:
            # Fallback logging if logger is not available
            print(f"Error running online learning: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

# Add other commands
cli.add_command(show_command)
cli.add_command(save_command)
cli.add_command(online_learning_command)

if __name__ == '__main__':
    cli() 
   
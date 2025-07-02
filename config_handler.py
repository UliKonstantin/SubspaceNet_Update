#!/usr/bin/env python3
"""
SubspaceNet Configuration Handler

This module provides a centralized way to handle configuration loading, validation,
and component creation for the SubspaceNet framework.
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import datetime
import re

from config.schema import Config
from config.loader import load_config, apply_overrides
from config.factory import create_components_from_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SubspaceNet.config')

def setup_configuration(
    config_path: str,
    output_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Tuple[Config, Dict[str, Any], Path]:
    """
    Set up configuration and create necessary components.
    
    Args:
        config_path: Path to the configuration file
        output_dir: Optional output directory for results
        overrides: Optional list of configuration overrides
        
    Returns:
        Tuple containing:
        - Validated configuration object
        - Dictionary of created components
        - Path to output directory
    """
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        # Load configuration
        config_obj = load_config(config_path)
        
        # Apply overrides if any
        if overrides:
            logger.info(f"Applying {len(overrides)} configuration overrides")
            config_obj = apply_overrides(config_obj, overrides)
        
        # Create a rich simulation name that includes key parameters
        if hasattr(config_obj, 'training') and hasattr(config_obj.training, 'simulation_name') and config_obj.training.simulation_name:
            base_name = config_obj.training.simulation_name
            
            # Extract key parameters
            params = []
            
            # Check if model parameters are specified - add if not already in name
            if hasattr(config_obj, 'model'):
                if hasattr(config_obj.model, 'type') and config_obj.model.type not in base_name:
                    params.append(config_obj.model.type)
                
                if hasattr(config_obj.model, 'params') and hasattr(config_obj.model.params, 'diff_method'):
                    diff_method = config_obj.model.params.diff_method
                    if isinstance(diff_method, str) and diff_method not in base_name:
                        params.append(diff_method)
            
            # System model parameters
            if hasattr(config_obj, 'system_model'):
                # Add N (number of antennas) if not already in name
                if hasattr(config_obj.system_model, 'N') and f"N{config_obj.system_model.N}" not in base_name:
                    params.append(f"N{config_obj.system_model.N}")
                
                # Add M (number of sources) if not already in name
                if hasattr(config_obj.system_model, 'M') and f"M{config_obj.system_model.M}" not in base_name:
                    params.append(f"M{config_obj.system_model.M}")
                
                # Add SNR if not already in name
                if hasattr(config_obj.system_model, 'snr') and f"SNR{config_obj.system_model.snr}" not in base_name:
                    params.append(f"SNR{config_obj.system_model.snr}")
                
                # Add field type if not already in name
                if hasattr(config_obj.system_model, 'field_type') and config_obj.system_model.field_type.title() not in base_name:
                    params.append(config_obj.system_model.field_type.title())
                
                # Add eta (calibration error) if non-zero and not already in name
                if hasattr(config_obj.system_model, 'eta') and config_obj.system_model.eta > 0 and f"eta{config_obj.system_model.eta}" not in base_name:
                    params.append(f"eta{config_obj.system_model.eta}")
            
            # Add subspace method (classic method) if specified
            if hasattr(config_obj, 'simulation') and hasattr(config_obj.simulation, 'subspace_methods'):
                methods = config_obj.simulation.subspace_methods
                if methods and len(methods) > 0:
                    # Use only the first method if multiple are specified to keep name reasonable
                    method_str = methods[0]
                    if method_str and method_str not in base_name:
                        params.append(method_str)
            
            # Add trajectory type if enabled
            if hasattr(config_obj, 'trajectory') and hasattr(config_obj.trajectory, 'enabled') and config_obj.trajectory.enabled:
                if hasattr(config_obj.trajectory, 'trajectory_type'):
                    traj_type = config_obj.trajectory.trajectory_type
                    if traj_type and traj_type not in base_name:
                        params.append(f"traj_{traj_type}")
            
            # Don't append if no extra parameters are needed
            if params:
                # Only add parameters if they're not already in the base name
                enhanced_name = f"{base_name}_{'_'.join(params)}"
                logger.info(f"Enhanced simulation name with parameters: {enhanced_name}")
                
                # Update the config object with the enhanced name
                config_obj.training.simulation_name = enhanced_name
        
        # Determine output directory in this priority:
        # 1. Command line output_dir (highest priority)
        # 2. simulation_name from config.training
        # 3. Simulation output_dir from config (with variable substitution)
        # 4. Fallback to timestamp-based name
        
        if output_dir is not None:
            # Command line output dir takes precedence
            output_path = Path(output_dir)
            logger.info(f"Using command-line specified output directory: {output_path}")
        elif hasattr(config_obj, 'training') and hasattr(config_obj.training, 'simulation_name') and config_obj.training.simulation_name:
            # Use simulation_name directly
            output_path = Path(f"experiments/results/{config_obj.training.simulation_name}")
            logger.info(f"Using simulation_name for output directory: {output_path}")
        elif hasattr(config_obj, 'simulation') and hasattr(config_obj.simulation, 'output_dir'):
            # Use config-specified output dir if available and process variables
            output_dir_str = config_obj.simulation.output_dir
            
            # Handle ${var} variable substitution
            var_pattern = r'\${([^}]+)}'
            matches = re.findall(var_pattern, output_dir_str)
            
            for match in matches:
                parts = match.split('.')
                if len(parts) == 2:
                    section, param = parts
                    if hasattr(config_obj, section) and hasattr(getattr(config_obj, section), param):
                        value = getattr(getattr(config_obj, section), param)
                        if value:
                            output_dir_str = output_dir_str.replace(f"${{{match}}}", str(value))
            
            output_path = Path(output_dir_str)
            logger.info(f"Using config output_dir with substitution: {output_path}")
        else:
            # Fall back to timestamp-based directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            output_path = Path(f"experiments/results/{timestamp}_{config_name}")
            logger.info(f"Using timestamp-based output directory: {output_path}")
        
        # Ensure the directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        # Create components
        logger.info("Creating components from configuration")
        components = create_components_from_config(config_obj)
        
        return config_obj, components, output_path
        
    except Exception as e:
        logger.error(f"Error in configuration setup: {e}", exc_info=True)
        raise 

def update_components_for_sweep(
    components: Dict[str, Any],
    config: Config,
    sweep_param: str,
    sweep_value: Any
) -> Dict[str, Any]:
    """
    Update components based on sweep parameter changes.
    Only recreates components that depend on the swept parameter.
    
    Args:
        components: Current components dictionary
        config: Configuration object
        sweep_param: Name of parameter being swept
        sweep_value: New value for the parameter
        
    Returns:
        Updated components dictionary
    """
    logger.info(f"Updating components for sweep parameter {sweep_param}={sweep_value}")
    
    # Map of parameters to components that need to be recreated
    param_to_components = {
        'eta': ['system_model'],  # eta affects system model and trajectory handler
        'snr': ['system_model'],  # snr affects system model and trajectory handler
        'M': ['system_model'],    # M affects system model and trajectory handler

    }
    
    # Get components that need to be updated for this parameter
    components_to_update = param_to_components.get(sweep_param.lower(), [])
    
    if not components_to_update:
        logger.info(f"No components need to be updated for {sweep_param}")
        return components
    
    # Create a copy of components to modify
    updated_components = components.copy()
    
    # Update each affected component
    for component_name in components_to_update:
        logger.info(f"Recreating {component_name} for {sweep_param}={sweep_value}")
        
        if component_name == 'system_model':
            # Recreate system model with new parameter
            from config.factory import create_system_model
            updated_components['system_model'] = create_system_model(config)
            
            # If model depends on system model, recreate it too
            if 'model' in components:
                from config.factory import create_model
                updated_components['model'] = create_model(config, updated_components['system_model'])
            
    
    return updated_components 
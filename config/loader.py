"""
Configuration loader for SubspaceNet.

This module provides functions for loading, validating, and managing configurations
for the SubspaceNet project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .schema import Config


def load_config(config_file_path: str) -> Config:
    """
    Load configuration from a YAML file and validate it.
    
    Args:
        config_file_path: Path to the configuration file
        
    Returns:
        Validated configuration object
    """
    config_path = Path(config_file_path)
    
    # If the specified config file doesn't exist, use the default config
    if not config_path.exists():
        default_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
        with open(default_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    
    # Validate configuration using Pydantic
    config = Config(**config_dict)
    
    return config


def save_config(config: Config, output_path: str) -> None:
    """
    Save a configuration to a YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Path where to save the configuration
    """
    config_dict = config.dict()
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def apply_overrides(config: Config, overrides: List[str]) -> Config:
    """
    Apply command-line overrides to the configuration.
    
    Args:
        config: Configuration object to modify
        overrides: List of override strings in the format "key=value"
        
    Returns:
        Modified configuration object
    """
    config_dict = config.dict()
    
    for override in overrides:
        key, value = override.split('=', 1)
        keys = key.split('.')
        
        # Navigate to the correct section in the configuration
        current = config_dict
        for k in keys[:-1]:
            if k not in current:
                raise ValueError(f"Invalid configuration key: {key}")
            current = current[k]
            
        # Convert value to the appropriate type
        final_key = keys[-1]
        if final_key not in current:
            raise ValueError(f"Invalid configuration key: {key}")
            
        # Get the current value to determine the type
        current_value = current[final_key]
        
        # Special handling for regularization parameter
        if key == "model.params.regularization":
            valid_values = [None, "aic", "mdl", "threshold", "null", "none"]
            if value.lower() not in [str(v).lower() for v in valid_values]:
                raise ValueError(f"Invalid value for regularization: {value}. "
                               f"Must be one of {valid_values}")
                
            # Normalize to None or the correct string
            if value.lower() in ["null", "none"]:
                current[final_key] = "null"
            else:
                current[final_key] = value.lower()
            continue
        
        # Rest of the conversion logic remains unchanged
        # Convert the string value to the appropriate type
        if current_value is None:
            if value.lower() == 'null' or value.lower() == 'none':
                current[final_key] = None
            else:
                current[final_key] = value
        elif isinstance(current_value, bool):
            current[final_key] = value.lower() == 'true'
        elif isinstance(current_value, int):
            current[final_key] = int(value)
        elif isinstance(current_value, float):
            current[final_key] = float(value)
        elif isinstance(current_value, list):
            # Parse as a comma-separated list
            if value.lower() == 'null' or value.lower() == 'none':
                current[final_key] = None
            else:
                current[final_key] = [item.strip() for item in value.split(',')]
        else:
            current[final_key] = value
    
    # Re-validate the configuration
    return Config(**config_dict) 
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
        
        # Create output directory with timestamp if not specified
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            output_path = Path(f"experiments/results/{timestamp}_{config_name}")
        else:
            output_path = Path(output_dir)
        
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
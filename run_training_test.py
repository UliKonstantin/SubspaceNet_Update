#!/usr/bin/env python3
"""
SubspaceNet Training Test Script

This script handles configuration setup for training tests using the SubspaceNet framework.
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import List, Optional
import datetime

from config.schema import Config
from config.loader import load_config, apply_overrides
from config.factory import create_components_from_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SubspaceNet')

@click.command()
@click.option('--config', '-c', default='configs/train_test_config.yaml',
              help='Path to the configuration file')
@click.option('--output', '-o', default=None,
              help='Output directory for experiment results (default: auto-generated in experiments/results)')
@click.option('--override', '-O', multiple=True,
              help='Override configuration parameter (format: key=value)')
def main(config: str, output: Optional[str], override: List[str]):
    """Run training test configuration setup."""
    logger.info(f"Loading configuration from {config}")
    
    try:
        # Load configuration
        config_obj = load_config(config)
        
        # Apply overrides if any
        if override:
            logger.info(f"Applying {len(override)} configuration overrides")
            config_obj = apply_overrides(config_obj, override)
        
        # Create output directory with timestamp if not specified
        if output is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config_name = os.path.splitext(os.path.basename(config))[0]
            output_dir = Path(f"experiments/results/{timestamp}_{config_name}")
        else:
            output_dir = Path(output)
        
        # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Create components
        logger.info("Creating components from configuration")
        components = create_components_from_config(config_obj)
        
        return config_obj, components, output_dir
        
    except Exception as e:
        logger.error(f"Error in configuration setup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 
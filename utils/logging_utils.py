"""
Logging utilities for SubspaceNet.

This module provides functions to configure logging based on configuration settings.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from config.schema import LoggingConfig


def setup_logging_from_config(logging_config: LoggingConfig, output_dir: Optional[Path] = None):
    """
    Setup logging based on configuration.
    
    Args:
        logging_config: LoggingConfig instance from the main config
        output_dir: Optional output directory for log files
    """
    # Convert string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Set global logging level
    global_level = level_map.get(logging_config.level, logging.INFO)
    logging.getLogger().setLevel(global_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(global_level)
    console_handler.setFormatter(formatter)
    
    # Add console handler to root logger if not already present
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
    
    # Setup file handler if requested
    if logging_config.log_to_file:
        if logging_config.log_file_path:
            log_file_path = Path(logging_config.log_file_path)
        elif output_dir:
            log_file_path = output_dir / "subspacenet.log"
        else:
            log_file_path = Path("subspacenet.log")
        
        # Ensure directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path)
        file_level = level_map.get(logging_config.log_file_level, global_level)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to root logger
        root_logger.addHandler(file_handler)
        
        print(f"Logging to file: {log_file_path}")
    
    # Set specific logger levels
    if logging_config.subspace_net_level:
        subspace_level = level_map.get(logging_config.subspace_net_level, global_level)
        logging.getLogger("SubspaceNet").setLevel(subspace_level)
    
    if logging_config.kalman_filter_level:
        kalman_level = level_map.get(logging_config.kalman_filter_level, global_level)
        logging.getLogger("SubspaceNet.kalman_filter").setLevel(kalman_level)
        logging.getLogger("SubspaceNet.kalman_filter.extended").setLevel(kalman_level)
        logging.getLogger("SubspaceNet.kalman_filter.models").setLevel(kalman_level)
    
    if logging_config.torch_level:
        torch_level = level_map.get(logging_config.torch_level, logging.WARNING)
        logging.getLogger("torch").setLevel(torch_level)
    
    if logging_config.matplotlib_level:
        matplotlib_level = level_map.get(logging_config.matplotlib_level, logging.WARNING)
        logging.getLogger("matplotlib").setLevel(matplotlib_level)
    
    # Log the configuration
    logger = logging.getLogger("SubspaceNet.logging")
    logger.info(f"Logging configured - Global level: {logging_config.level}")
    if logging_config.subspace_net_level:
        logger.info(f"SubspaceNet level: {logging_config.subspace_net_level}")
    if logging_config.kalman_filter_level:
        logger.info(f"Kalman filter level: {logging_config.kalman_filter_level}")
    if logging_config.log_to_file:
        logger.info(f"File logging enabled: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience functions for quick logging level changes
def set_verbose_logging():
    """Enable verbose logging (DEBUG level)."""
    logging.getLogger().setLevel(logging.DEBUG)
    print("Verbose logging enabled (DEBUG level)")

def set_normal_logging():
    """Enable normal logging (INFO level)."""
    logging.getLogger().setLevel(logging.INFO)
    print("Normal logging enabled (INFO level)")

def set_quiet_logging():
    """Enable quiet logging (WARNING level)."""
    logging.getLogger().setLevel(logging.WARNING)
    print("Quiet logging enabled (WARNING level)")

def set_minimal_logging():
    """Enable minimal logging (ERROR level)."""
    logging.getLogger().setLevel(logging.ERROR)
    print("Minimal logging enabled (ERROR level)")

"""
Kalman Filter implementations for tracking in DOA estimation.

This package provides Kalman Filter implementations for tracking state
variables like angle and range in direction-of-arrival (DOA) estimation.
"""

import logging
from config.schema import TrajectoryType
from simulation.kalman_filter.base import KalmanFilter1D
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D
from simulation.kalman_filter.batch import BatchKalmanFilter1D
from simulation.kalman_filter.batch_extended import BatchExtendedKalmanFilter1D

logger = logging.getLogger("SubspaceNet.kalman_filter")

def get_kalman_filter(config, trajectory_type=None):
    """
    Get appropriate Kalman Filter based on config and trajectory type.
    
    This function selects either the standard KalmanFilter1D or the 
    ExtendedKalmanFilter1D based on the configuration and trajectory type.
    
    Args:
        config: Configuration object with kalman_filter and trajectory settings
        trajectory_type: Type of trajectory to model (defaults to config value)
        
    Returns:
        KalmanFilter1D or ExtendedKalmanFilter1D instance
    """
    # Get trajectory type from config if not specified
    if trajectory_type is None:
        trajectory_type = config.trajectory.trajectory_type
    
    # Check if filter_type is explicitly specified
    filter_type = getattr(config.kalman_filter, "filter_type", "standard")
    
    # Determine which filter to use
    use_extended = False
    
    # Use extended filter if explicitly requested
    if filter_type.lower() == "extended":
        use_extended = True
        logger.info("Using ExtendedKalmanFilter as specified in config")
    
    # Use extended filter for non-linear trajectory types
    elif trajectory_type in [
        TrajectoryType.SINE_ACCEL_NONLINEAR, 
        TrajectoryType.MULT_NOISE_NONLINEAR
    ]:
        use_extended = True
        logger.info(f"Using ExtendedKalmanFilter for non-linear trajectory type: {trajectory_type}")
    
    # Create and return the appropriate filter
    if use_extended:
        return ExtendedKalmanFilter1D.create_from_config(config, trajectory_type)
    else:
        # Use the original KalmanFilter1D for simple models
        return KalmanFilter1D.create_from_config(config)

def get_batch_extended_kalman_filter(config, batch_size, max_sources, device=None, trajectory_type=None):
    """
    Get BatchExtendedKalmanFilter1D from configuration.
    
    Args:
        config: Configuration object containing kalman_filter and trajectory settings
        batch_size: Number of trajectories in the batch
        max_sources: Maximum number of sources
        device: PyTorch device
        trajectory_type: Type of trajectory to model (defaults to config value)
        
    Returns:
        BatchExtendedKalmanFilter1D instance
    """
    return BatchExtendedKalmanFilter1D.from_config(config, batch_size, max_sources, device, trajectory_type)


def get_batch_kalman_filter(config, batch_size, max_sources, device=None):
    """
    Get BatchKalmanFilter1D from configuration.
    
    Args:
        config: Configuration object
        batch_size: Number of trajectories in the batch
        max_sources: Maximum number of sources per trajectory
        device: PyTorch device
        
    Returns:
        BatchKalmanFilter1D instance
    """
    return BatchKalmanFilter1D.from_config(config, batch_size, max_sources, device)


__all__ = [
    'KalmanFilter1D',
    'BatchKalmanFilter1D',
    'ExtendedKalmanFilter1D',
    'BatchExtendedKalmanFilter1D',
    'get_kalman_filter',
    'get_batch_kalman_filter',
    'get_batch_extended_kalman_filter'
] 
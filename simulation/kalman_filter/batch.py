"""
Batch Kalman Filter implementation for efficient processing of multiple trajectories.

This module provides a batch-enabled Kalman Filter implementation for tracking
multiple sources across multiple trajectories efficiently.
"""

import numpy as np
import torch
import logging

logger = logging.getLogger("SubspaceNet.kalman_filter.batch")

class BatchKalmanFilter1D:
    """
    Batch-enabled 1D Kalman Filter for tracking multiple sources across multiple trajectories.
    
    This extends the KalmanFilter1D concept to work efficiently with batches of data
    by vectorizing the Kalman Filter operations.
    
    Notes:
        - Uses PyTorch tensors for efficient batch operations
        - Can handle variable number of sources per trajectory using masks
    """
    
    def __init__(self, batch_size, max_sources, Q, R, P0=1.0, device=None):
        """
        Initialize the batch Kalman Filter.
        
        Args:
            batch_size: Number of trajectories in the batch
            max_sources: Maximum number of sources per trajectory
            Q: Process noise variance (scalar or batch)
            R: Measurement noise variance (scalar or batch)
            P0: Initial state covariance (scalar or batch)
            device: PyTorch device (for GPU acceleration)
        """
        self.batch_size = batch_size
        self.max_sources = max_sources
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prevent exactly zero measurement noise (numerical stability)
        if isinstance(R, (int, float)) and R == 0:
            R = 1e-6
            
        # Convert scalars to tensors if needed
        if isinstance(Q, (int, float)):
            self.Q = torch.full((batch_size, max_sources), Q, device=self.device)
        else:
            self.Q = torch.as_tensor(Q, device=self.device)
            
        if isinstance(R, (int, float)):
            self.R = torch.full((batch_size, max_sources), R, device=self.device)
        else:
            self.R = torch.as_tensor(R, device=self.device)
            
        if isinstance(P0, (int, float)):
            self.P0 = torch.full((batch_size, max_sources), P0, device=self.device)
        else:
            self.P0 = torch.as_tensor(P0, device=self.device)
        
        # Initialize state and masks
        self.x = torch.zeros((batch_size, max_sources), device=self.device)  # State estimates
        self.P = self.P0.clone()  # State covariances
        self.state_initialized = torch.zeros((batch_size, max_sources), dtype=torch.bool, device=self.device)
        
        # Create source mask (to handle variable sources per trajectory)
        self.source_mask = torch.zeros((batch_size, max_sources), dtype=torch.bool, device=self.device)
        
        logger.info(f"Initialized Batch KF with batch_size={batch_size}, max_sources={max_sources} on {self.device}")
        
    @classmethod
    def from_config(cls, config, batch_size, max_sources, device=None):
        """
        Create BatchKalmanFilter1D from configuration.
        
        Args:
            config: Configuration object containing kalman_filter settings
            batch_size: Number of trajectories in the batch
            max_sources: Maximum number of sources
            device: PyTorch device
            
        Returns:
            BatchKalmanFilter1D instance
        """
        # Get parameters from config
        kf_process_noise_std_dev = config.kalman_filter.process_noise_std_dev
        if kf_process_noise_std_dev is None:
            kf_process_noise_std_dev = config.trajectory.random_walk_std_dev
            logger.debug(f"Using trajectory.random_walk_std_dev ({kf_process_noise_std_dev}) for KF process noise")
        
        # Calculate variances from standard deviations
        kf_Q = kf_process_noise_std_dev ** 2
        kf_R = config.kalman_filter.measurement_noise_std_dev ** 2
        kf_P0 = config.kalman_filter.initial_covariance
        
        logger.info(f"Batch Kalman Filter parameters: Q={kf_Q}, R={kf_R}, P0={kf_P0}")
        
        return cls(batch_size, max_sources, kf_Q, kf_R, kf_P0, device)
    
    def initialize_states(self, initial_states, num_sources):
        """
        Initialize the state estimates for multiple trajectories.
        
        Args:
            initial_states: Tensor of initial states [batch_size, max_sources]
            num_sources: Tensor or list with number of sources per trajectory [batch_size]
        """
        # Convert to tensor if needed
        if not isinstance(num_sources, torch.Tensor):
            num_sources = torch.tensor(num_sources, device=self.device)
            
        # Create source mask based on number of sources per trajectory
        self.source_mask.fill_(False)
        for i in range(self.batch_size):
            self.source_mask[i, :num_sources[i]] = True
        
        # Initialize states and mark as initialized
        self.x = initial_states.clone().to(self.device)
        self.P = self.P0.clone()
        self.state_initialized = self.source_mask.clone()
        
        logger.debug(f"Initialized batch states with shape {initial_states.shape}")
    
    def predict(self):
        """
        Perform the time update (prediction) step for all trajectories and sources.
        
        Returns:
            Tensor of predicted states [batch_size, max_sources]
        """
        # Only update states that have been initialized
        mask = self.state_initialized & self.source_mask
        
        # For random walk model, state prediction is the same
        # x_pred = x (state doesn't change in prediction for random walk)
        
        # Covariance prediction: P_pred = P + Q
        self.P[mask] = self.P[mask] + self.Q[mask]
        
        logger.debug(f"Batch prediction complete for {mask.sum().item()} states")
        return self.x
    
    def update(self, measurements, measurement_mask=None):
        """
        Perform the measurement update step for all trajectories and sources.
        
        Args:
            measurements: Tensor of measurements [batch_size, max_sources]
            measurement_mask: Boolean mask indicating valid measurements [batch_size, max_sources]
            
        Returns:
            Tuple of (updated states [batch_size, max_sources], error covariances [batch_size, max_sources])
        """
        if measurement_mask is None:
            measurement_mask = self.source_mask
        else:
            # Ensure we only update initialized and active sources
            measurement_mask = measurement_mask & self.state_initialized & self.source_mask
        
        # Only process valid measurements
        if measurement_mask.sum() == 0:
            return self.x, self.P
        
        # Innovation: y = z - x
        y = torch.zeros_like(self.x)
        y[measurement_mask] = measurements[measurement_mask] - self.x[measurement_mask]
        
        # Innovation covariance: S = P + R
        S = self.P.clone()
        S[measurement_mask] = S[measurement_mask] + self.R[measurement_mask]
        
        # Kalman gain: K = P / S
        K = torch.zeros_like(self.P)
        K[measurement_mask] = self.P[measurement_mask] / S[measurement_mask]
        
        # State update: x_new = x + K * y
        x_new = self.x.clone()
        x_new[measurement_mask] = self.x[measurement_mask] + K[measurement_mask] * y[measurement_mask]
        
        # Covariance update: P_new = (1 - K) * P
        P_new = self.P.clone()
        P_new[measurement_mask] = (1 - K[measurement_mask]) * self.P[measurement_mask]
        
        # Update state and covariance
        self.x = x_new
        self.P = P_new
        
        logger.debug(f"Batch update complete for {measurement_mask.sum().item()} measurements")
        return self.x, self.P
    
    def predict_and_update(self, measurements, measurement_mask=None):
        """
        Perform both prediction and update steps in sequence.
        
        Args:
            measurements: Tensor of measurements [batch_size, max_sources]
            measurement_mask: Boolean mask indicating valid measurements [batch_size, max_sources]
            
        Returns:
            Tuple of (updated states [batch_size, max_sources], error covariances [batch_size, max_sources])
        """
        self.predict()
        return self.update(measurements, measurement_mask) 
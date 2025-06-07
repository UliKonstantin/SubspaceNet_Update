"""
Base Kalman Filter implementation for DOA tracking.

This module provides the original KalmanFilter1D implementation for
angle tracking in DOA estimation.
"""

import numpy as np
import logging

logger = logging.getLogger("SubspaceNet.kalman_filter.base")

class KalmanFilter1D:
    """
    Standard Kalman Filter for tracking a 1D state variable.
    
    This implements a constant velocity model with Gaussian noise:
    - State transition: x_k = x_{k-1} + w_k
    - Observation model: z_k = x_k + v_k
    - w_k ~ N(0, Q), v_k ~ N(0, R)
    """
    
    def __init__(self, Q, R, P0=1.0):
        """
        Initialize the Kalman Filter.
        
        Args:
            Q: Process noise variance (scalar)
            R: Measurement noise variance (scalar)
            P0: Initial state covariance (scalar)
        """
        # Prevent exactly zero process/measurement noise
        if Q == 0:
            Q = 1e-6
        if R == 0:
            R = 1e-6
            
        self.Q = Q  # Process noise variance
        self.R = R  # Measurement noise variance
        self.P0 = P0  # Initial state covariance
        
        # State will be initialized later
        self.x = None  # State estimate
        self.P = P0  # State estimate covariance
        
        logger.debug(f"Initialized Kalman Filter with Q={Q}, R={R}, P0={P0}")
    
    @classmethod
    def from_config(cls, config):
        """
        Create KalmanFilter1D parameters from configuration.
        
        Args:
            config: Configuration object with kalman_filter settings
            
        Returns:
            tuple: (Q, R, P0) - Parameters needed for KalmanFilter1D initialization
        """
        # Get process noise and measurement noise standard deviations
        if config.kalman_filter.process_noise_std_dev is None:
            # Use trajectory random walk standard deviation as process noise
            process_noise_std = config.trajectory.random_walk_std_dev
        else:
            process_noise_std = config.kalman_filter.process_noise_std_dev
            
        measurement_noise_std = config.kalman_filter.measurement_noise_std_dev
        
        # Convert standard deviations to variances
        kf_Q = process_noise_std ** 2
        kf_R = measurement_noise_std ** 2
        
        # Get initial covariance
        kf_P0 = config.kalman_filter.initial_covariance
        
        logger.info(f"Kalman Filter parameters: Q={kf_Q}, R={kf_R}, P0={kf_P0}")
        return kf_Q, kf_R, kf_P0
    
    def initialize_state(self, x0):
        """
        Initialize the state estimate.
        
        Args:
            x0: Initial state estimate (scalar)
        """
        self.x = x0
        self.P = self.P0
        logger.debug(f"Initialized state to x0={x0} with covariance P0={self.P0}")
    
    def predict(self):
        """
        Perform the time update (prediction) step.
        
        Returns:
            Predicted state (scalar)
        """
        if self.x is None:
            raise ValueError("State must be initialized before prediction")
        
        # State prediction (x_k|k-1 = F * x_k-1|k-1)
        # With F = 1 for random walk model
        x_pred = self.x
        
        # Covariance prediction (P_k|k-1 = F * P_k-1|k-1 * F^T + Q)
        # With F = 1 for random walk model
        P_pred = self.P + self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
        
        logger.debug(f"Prediction: x={x_pred}, P={P_pred}")
        return x_pred
    
    def update(self, z):
        """
        Perform the measurement update step.
        
        Args:
            z: Measurement (scalar)
            
        Returns:
            Updated state estimate (scalar)
        """
        if self.x is None:
            raise ValueError("State must be initialized before update")
        
        # Measurement model is identity (H = 1) for our problem
        # Innovation / measurement residual (y = z - H * x_k|k-1)
        y = z - self.x
        
        # Innovation covariance (S = H * P_k|k-1 * H^T + R)
        S = self.P + self.R
        
        # Kalman gain (K = P_k|k-1 * H^T * S^-1)
        K = self.P / S
        
        # State update (x_k|k = x_k|k-1 + K * y)
        x_new = self.x + K * y
        
        # Covariance update (P_k|k = (I - K * H) * P_k|k-1)
        P_new = (1 - K) * self.P
        
        # Update state and covariance
        self.x = x_new
        self.P = P_new
        
        logger.debug(f"Update with z={z}: y={y}, K={K}, x={x_new}, P={P_new}")
        return x_new

    @classmethod
    def create_from_config(cls, config):
        """
        Create KalmanFilter1D instance directly from config.
        
        Args:
            config: Configuration object with kalman_filter settings
            
        Returns:
            KalmanFilter1D instance
        """
        params = cls.from_config(config)
        return cls(Q=params[0], R=params[1], P0=params[2])
    
    @classmethod
    def create_filters_from_config(cls, config, num_instances=1):
        """
        Create multiple KalmanFilter1D instances directly from config.
        
        Args:
            config: Configuration object with kalman_filter settings
            num_instances: Number of filter instances to create
            
        Returns:
            list: List of KalmanFilter1D instances
        """
        params = cls.from_config(config)
        filters = [cls(Q=params[0], R=params[1], P0=params[2]) for _ in range(num_instances)]
        return filters 